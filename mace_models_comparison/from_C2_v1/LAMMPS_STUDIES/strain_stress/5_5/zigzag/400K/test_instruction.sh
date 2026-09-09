#!/bin/bash

(
	which lmp
	which python
	echo launching background process
	source test2.sh
)&
