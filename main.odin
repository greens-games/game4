package main

import "core:text/regex"
import "slow_works"
import "simd_version"
import "matrix_version"
import "graph"
import "constants"

import "core:fmt"
import "core:mem"
import "core:math/rand"
import rl "vendor:raylib"

Version :: enum {
	SIMD,
	SLOW,
	MATRIX,
}

main :: proc() {
	rand.reset(1)
	gui: bool = false
	if !gui {
		version:Version = .SLOW
		switch version {
		case .SIMD:
			simd_version.run()
		case .SLOW:
			slow_works.run()
		case .MATRIX:
			matrix_version.run()
		}
	} else {

		when ODIN_DEBUG {
			track: mem.Tracking_Allocator
			mem.tracking_allocator_init(&track, context.allocator)
			context.allocator = mem.tracking_allocator(&track)
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
		
		input_matrix, expected_vector := slow_works.generate_input() 
		defer {
			slow_works.clean_matrix(input_matrix)
		}

		layers := slow_works.init_network()
		defer {
			for &layer in layers {
				slow_works.clean_matrix(layer.weights)
			}
		}
		//run gui and draw network
		rl.InitWindow(920,680,"Network")
		
		nodes: [4]graph.Node
		layer1_neurons: []f64
		output_neurons: [4]f64
		input: []f64
		input_layer: [4]graph.Node
		hidden_nodes: [4]graph.Node
		output_nodes: [4]graph.Node
		idx := 0
		for !rl.WindowShouldClose() {
			if rl.IsKeyPressed(.SPACE) {
				if idx < constants.ITERATIONS {
					input_layer = graph.convert_to_nodes(input_matrix[idx], .NONE)
					input = slow_works.normalize_vector(input_matrix[idx])
					layer1_neurons = slow_works.forward_hidden_layer(input, layers[0])
					output_neurons = slow_works.forward_output(layer1_neurons, layers[1])
					hidden_nodes = graph.convert_to_nodes(layer1_neurons[:], .RELU)
					output_nodes = graph.convert_to_nodes(output_neurons[:], .SOFT_MAX)
					idx += 1
				} else {
					fmt.println("AT MAX ITERATIONS")
				}
			}
			rl.BeginDrawing()
			rl.ClearBackground(rl.BLACK)
			graph.draw_input_layer(input_layer[:])
			graph.draw_hidden_layers(hidden_nodes [:])
			graph.draw_output_layer(output_nodes[:])
			rl.EndDrawing()
			free_all()
		}
	}
}
