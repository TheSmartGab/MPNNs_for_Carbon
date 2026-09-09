#!/bin/bash

checkpoint_path="best.ckpt"
package_path="packaged_model.nequip.zip"
output_path="deployed_pair.nequip.pth"

nequip-package build "${checkpoint_path}" "${package_path}"
nequip-compile --mode torchscript --device cpu --target pair_nequip "${package_path}" "${output_path}"
