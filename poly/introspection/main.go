package main

import (
	"fmt"
	"github.com/openfluke/loom/poly"
)

func main() {
	fmt.Println("=== M-POLY-VTD VolumetricNetwork Introspection Demo ===")

	n := poly.NewVolumetricNetwork(1, 1, 1, 1)

	// 1. List all public methods
	fmt.Println("\n[1] Listing Public Methods:")
	methods := n.ListMethods()
	for _, m := range methods {
		fmt.Printf(" - %s\n", m)
	}

	// 2. Check for specific methods
	fmt.Println("\n[2] Checking Specific Methods:")
	target := "GetLayer"
	if n.HasMethod(target) {
		fmt.Printf(" - Method '%s' exists.\n", target)
		
		sig, err := n.GetMethodSignature(target)
		if err == nil {
			fmt.Printf("   Signature: %s\n", sig)
		}
	}

	// 3. Output full metadata JSON
	fmt.Println("\n[3] Full Methods Metadata (JSON):")
	jsonStr, err := n.GetMethodsJSON()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println(jsonStr)
	}

	fmt.Println("\n=== Introspection Demo Complete ===")
}
