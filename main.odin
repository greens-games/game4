package main

import "core:fmt"
import "core:mem"
import "core:math"

main :: proc() {

	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not free: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}
	input := [3]f32 {1.,0.,2.}
	w1 := [3]f32 {2,3,6}
	bias:f32 = 5.
	z := weighted_sum(input[:], w1[:]) + bias
	/* layer1 := create_matrix(3,1)
	layer1[0][0] = 2
	layer1[1][0] = 3
	layer1[2][0] = 6 */
	// rows must = length of input vector 
	//cols = number of neurons can be anything 
	//Odin is limited ot 4x4 builtin 
	layer1: matrix[3,2]f32 
	layer1 = {
		2, 5,
		3, 7,
		6, 1,
	}

	layer2: matrix[2,2]f32 
	layer2 = {
		2, 5,
		3, 7,
	}

	fmt.println(input * layer1)

}

weighted_sum :: proc(input, weights: []f32) -> f32 {
	assert(len(input) == len(weights))
	sum:f32 = 0.
	for i in 0..<len(input) {
		sum += input[i] * weights[i]
	}
	return sum
}

relu :: proc(z: f32) -> f32 {
	return max(0., z)
}

soft_max :: proc() {
}

create_matrix :: proc(rows, cols: int) -> [][]f32 {
	m := make_slice([][]f32 , rows)
	for &row in m {
		row = make_slice([]f32, cols)
	}
	return m 
}
