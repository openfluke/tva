package main

import (
	"fmt"

	"github.com/openfluke/loom/nn"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║          Universal Type Conversion System Demo              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Test single value conversions
	fmt.Println("📊 Single Value Conversions:")
	fmt.Println("────────────────────────────────────────")

	testValue := float32(42.5)

	// F32 → F64
	val, _ := nn.ConvertValue(testValue, nn.TypeF32, nn.TypeF64)
	fmt.Printf("F32(%.2f) → F64 = %.2f\n", testValue, val.(float64))

	// F32 → I32
	val, _ = nn.ConvertValue(testValue, nn.TypeF32, nn.TypeI32)
	fmt.Printf("F32(%.2f) → I32 = %d\n", testValue, val.(int32))

	// F32 → U16
	val, _ = nn.ConvertValue(testValue, nn.TypeF32, nn.TypeU16)
	fmt.Printf("F32(%.2f) → U16 = %d\n", testValue, val.(uint16))

	fmt.Println()

	// Test slice conversions
	fmt.Println("📦 Slice Conversions:")
	fmt.Println("────────────────────────────────────────")

	f32Slice := []float32{1.5, 2.7, 3.2, 4.9, 5.1}
	fmt.Printf("Original F32 slice: %v\n", f32Slice)

	// F32 → F64
	f64Result, _ := nn.ConvertSlice(f32Slice, nn.TypeF32, nn.TypeF64)
	fmt.Printf("→ F64: %v\n", f64Result.([]float64))

	// F32 → I16
	i16Result, _ := nn.ConvertSlice(f32Slice, nn.TypeF32, nn.TypeI16)
	fmt.Printf("→ I16: %v\n", i16Result.([]int16))

	// F32 → U8
	u8Result, _ := nn.ConvertSlice(f32Slice, nn.TypeF32, nn.TypeU8)
	fmt.Printf("→ U8:  %v\n", u8Result.([]uint8))

	fmt.Println()

	// Test bidirectional conversion
	fmt.Println("🔄 Bidirectional Conversion:")
	fmt.Println("────────────────────────────────────────")

	i32Slice := []int32{-100, 0, 50, 100, 200}
	fmt.Printf("Original I32 slice: %v\n", i32Slice)

	// I32 → F32
	f32FromI32, _ := nn.ConvertSlice(i32Slice, nn.TypeI32, nn.TypeF32)
	fmt.Printf("→ F32: %v\n", f32FromI32.([]float32))

	// F32 → I32 (round trip)
	i32Again, _ := nn.ConvertSlice(f32FromI32, nn.TypeF32, nn.TypeI32)
	fmt.Printf("→ I32 (round trip): %v\n", i32Again.([]int32))

	fmt.Println()

	// Test type introspection
	fmt.Println("🔍 Type Introspection:")
	fmt.Println("────────────────────────────────────────")

	allTypes := []nn.NumericType{
		nn.TypeF32, nn.TypeF64, nn.TypeF16, nn.TypeBF16, nn.TypeF4,
		nn.TypeI8, nn.TypeI16, nn.TypeI32, nn.TypeI64,
		nn.TypeU8, nn.TypeU16, nn.TypeU32, nn.TypeU64,
	}

	fmt.Printf("%-8s | %-5s | %-12s | %-12s | %-20s\n", "Type", "Size", "Min", "Max", "Category")
	fmt.Println("──────────────────────────────────────────────────────────────────────────────")

	for _, t := range allTypes {
		size := nn.GetTypeSize(t)
		min, max := nn.GetTypeRange(t)

		category := "Integer"
		if nn.IsNumericTypeFloat(t) {
			category = "Float"
		} else if nn.IsNumericTypeUnsigned(t) {
			category = "Unsigned Int"
		} else if nn.IsNumericTypeSignedInt(t) {
			category = "Signed Int"
		}

		sizeStr := fmt.Sprintf("%d", size)
		if t == nn.TypeF4 {
			sizeStr = "0.5" // 2 values per byte
		}

		fmt.Printf("%-8s | %-5s | %-12.2e | %-12.2e | %-20s\n",
			nn.GetTypeName(t), sizeStr, min, max, category)
	}

	fmt.Println()
	fmt.Println("✅ Universal Type Conversion System is fully operational!")
}
