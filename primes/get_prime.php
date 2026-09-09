#!/usr/bin/env php
<?php
// Make script executable and runnable from anywhere

// Directory where this script lives
$SCRIPT_DIR = dirname(__FILE__);

// Load the primes array (assumes primes.php is in the same folder)
require $SCRIPT_DIR . '/primes.php';

// Optional: path to the log file
$LOG_FILE = $SCRIPT_DIR . '/log_100000.txt';

// Get the index from command line
$index = $argv[1] ?? null;

if ($index === null || !isset($data[$index])) {
    fwrite(STDERR, "Index not found or missing.\n");
    exit(1);
}

// Output the prime at that index
echo $data[$index] . "\n";
