import os
import sys
import argparse
from ase.io import read, write

def find_frame_offset(f, target_ts, filesize):
    """Binary search to find the byte offset of the target timestep."""
    low = 0
    high = filesize
    
    while low <= high:
        mid = (low + high) // 2
        f.seek(mid)
        f.readline()  # Align to the next full line
        
        found_ts = None
        current_offset = f.tell()
        
        # Scan forward slightly to find the next header
        while current_offset < filesize:
            line = f.readline()
            if b"ITEM: TIMESTEP" in line:
                header_offset = current_offset # Position of "ITEM: TIMESTEP"
                ts_line = f.readline().strip()
                if ts_line:
                    found_ts = int(ts_line)
                    break
            current_offset = f.tell()
        
        if found_ts is None or found_ts > target_ts:
            high = mid - 1
        elif found_ts < target_ts:
            low = mid + 1
        else:
            return header_offset # Found the start of the frame
    return None

def frame_generator(f, start_offset):
    """Yields lines for a single frame to avoid loading the whole file."""
    f.seek(start_offset)
    # Yield the first header we already found
    yield f.readline().decode('utf-8') 
    
    while True:
        line = f.readline().decode('utf-8')
        if not line or "ITEM: TIMESTEP" in line:
            break
        yield line

def main():
    parser = argparse.ArgumentParser(description="O(log n) extraction and conversion.")
    parser.add_argument("input", help="LAMMPS text dump file")
    parser.add_argument("timestep", type=int, help="Target timestep")
    parser.add_argument("output", help="Output lammps-data file")
    args = parser.parse_args()

    filesize = os.path.getsize(args.input)
    
    with open(args.input, 'rb') as f:
        offset = find_frame_offset(f, args.timestep, filesize)
        
        if offset is None:
            print(f"Error: Timestep {args.timestep} not found.")
            sys.exit(1)
        
        # Wrap our generator so ASE thinks it's reading a file
        from io import StringIO
        frame_data = StringIO("".join(frame_generator(f, offset)))
        
        # Read from memory-like object and write
        atoms = read(frame_data, format='lammps-dump-text')
        write(args.output, atoms, format='lammps-data', velocities=True)
        print(f"Done. Frame {args.timestep} saved to {args.output}")

if __name__ == "__main__":
    main()