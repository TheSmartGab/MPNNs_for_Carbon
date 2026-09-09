def agnesi(r, r0, a, q, p):

    ratio=r/r0
    return 1 / (1 + a * (ratio)**q / (1 + (ratio)**(q-p)))
