#!/bin/bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
go run main.go --layer=Dense --dtype=float32 --depth=deep
