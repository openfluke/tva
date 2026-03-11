package main

import (
	"fmt"
	"log"
	"github.com/openfluke/webgpu/wgpu"
)

func main() {
	instance := wgpu.CreateInstance(nil)
	if instance == nil {
		log.Fatalf("failed to create WGPU instance")
	}
	defer instance.Release()

	adapter, err := instance.RequestAdapter(&wgpu.RequestAdapterOptions{
		PowerPreference: wgpu.PowerPreferenceHighPerformance,
	})
	if err != nil {
		log.Fatalf("failed to request adapter: %v", err)
	}
	defer adapter.Release()

	info := adapter.GetInfo()
	fmt.Printf("Adapter: %s (Backend: %v)\n", info.Name, info.BackendType)

	adapterLimits := adapter.GetLimits()
	al := adapterLimits.Limits
	fmt.Printf("Adapter Limits:\n")
	fmt.Printf("  MaxStorageBufferBindingSize: %d MB\n", al.MaxStorageBufferBindingSize / (1024*1024))
	fmt.Printf("  MaxBufferSize: %d MB\n", al.MaxBufferSize / (1024*1024))
	fmt.Printf("  MaxComputeWorkgroupStorageSize: %d\n", al.MaxComputeWorkgroupStorageSize)
	fmt.Printf("  MaxComputeInvocationsPerWorkgroup: %d\n", al.MaxComputeInvocationsPerWorkgroup)
	fmt.Printf("  MaxComputeWorkgroupsPerDimension: %d\n", al.MaxComputeWorkgroupsPerDimension)
	fmt.Printf("  MaxBindGroups: %d\n", al.MaxBindGroups)

	// Attempt 1: Current approach (Partial request, most at 0)
	targetStorage := al.MaxStorageBufferBindingSize
	if targetStorage > 512*1024*1024 {
		targetStorage = 512 * 1024 * 1024
	}

	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{
			Limits: wgpu.Limits{
				MaxStorageBufferBindingSize:       targetStorage,
				MaxBufferSize:                     al.MaxBufferSize,
				MaxComputeWorkgroupStorageSize:    al.MaxComputeWorkgroupStorageSize,
				MaxComputeInvocationsPerWorkgroup: al.MaxComputeInvocationsPerWorkgroup,
				MaxComputeWorkgroupsPerDimension:  al.MaxComputeWorkgroupsPerDimension,
			},
		},
	})
	if err != nil {
		fmt.Printf("Attempt 1 (Partial) Failed: %v\n", err)
	} else {
		fmt.Println("Attempt 1 Succeeded!")
		device.Release()
	}

	// Attempt 2: Full copy
	device2, err2 := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{
			Limits: al,
		},
	})
	if err2 != nil {
		fmt.Printf("Attempt 2 (Full Copy) Failed: %v\n", err2)
	} else {
		fmt.Println("Attempt 2 Succeeded!")
		device2.Release()
	}

	// Attempt 3: Full copy but specific cap on BufferSize to 4GB if it looks like "unlimited"
	required3 := al
	if required3.MaxBufferSize > 4*1024*1024*1024 {
		required3.MaxBufferSize = 4 * 1024 * 1024 * 1024
	}
	if required3.MaxStorageBufferBindingSize > 1*1024*1024*1024 {
		required3.MaxStorageBufferBindingSize = 1 * 1024 * 1024 * 1023 // a bit less than 1GB
	}

	// Attempt 3: Full copy but specific cap on BufferSize to 4GB if it looks like "unlimited"
	required3 = al
	if required3.MaxBufferSize > 4*1024*1024*1024 {
		required3.MaxBufferSize = 4 * 1024 * 1024 * 1024
	}
	if required3.MaxStorageBufferBindingSize > 1*1024*1024*1024 {
		required3.MaxStorageBufferBindingSize = 1 * 1024 * 1024 * 1023
	}

	device3, err3 := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{
			Limits: required3,
		},
	})
	if err3 != nil {
		fmt.Printf("Attempt 3 (Capped Large) Failed: %v\n", err3)
	} else {
		fmt.Println("Attempt 3 Succeeded!")
		device3.Release()
	}

	// Attempt 4: Nil descriptor (default limits)
	device4, err4 := adapter.RequestDevice(nil)
	var defaultLimits wgpu.Limits
	if err4 != nil {
		fmt.Printf("Attempt 4 (Nil) Failed: %v\n", err4)
	} else {
		fmt.Println("Attempt 4 Succeeded!")
		defaultLimits = device4.GetLimits().Limits
		device4.Release()
	}

	// Attempt 5: Default limits but increased storage limits
	req5 := defaultLimits
	req5.MaxStorageBufferBindingSize = 1 * 1024 * 1024 * 1024 // 1 GB
	req5.MaxBufferSize = 2 * 1024 * 1024 * 1024               // 2 GB

	device5, err5 := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{
			Limits: req5,
		},
	})
	if err5 != nil {
		fmt.Printf("Attempt 5 (Default + Storage Boost) Failed: %v\n", err5)
	} else {
		fmt.Println("Attempt 5 Succeeded!")
		device5.Release()
	}

	// Attempt 6: gpu/context.go exactly (clone adapter limits, cap storage and buffer to 1GB)
	req6 := adapter.GetLimits().Limits
	if req6.MaxStorageBufferBindingSize > 1024*1024*1024 {
		req6.MaxStorageBufferBindingSize = 1024 * 1024 * 1024
	}
	if req6.MaxBufferSize > 1024*1024*1024 {
		req6.MaxBufferSize = 1024 * 1024 * 1024
	}

	device6, err6 := adapter.RequestDevice(&wgpu.DeviceDescriptor{
		RequiredLimits: &wgpu.RequiredLimits{
			Limits: req6,
		},
	})
	if err6 != nil {
		fmt.Printf("Attempt 6 (gpu/context.go logic) Failed: %v\n", err6)
	} else {
		fmt.Println("Attempt 6 Succeeded!")
		device6.Release()
	}
}
