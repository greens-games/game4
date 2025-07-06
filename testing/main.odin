package main

import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:os"
import "core:strings"
import "core:strconv"
import "core:mem"

Pokemon_Type :: enum {
	FIRE,
	GRASS,
	WATER,
	FLYING,
	GROUND,
	ROCK,
	POISON,
	STEEL,
	ELECTRIC,
	FIGHTING,
	FAIRY,
	NORMAL,
	BUG,
	PSYCHIC,
	DARK,
	GHOST,
}

Pokemon_Stats :: struct {
	hp: f64,
	attack: f64,
	defense: f64,
	special_attack: f64,
	special_defense: f64,
	speed: f64,
}
Pokemon :: struct {
	id: int,
	name: string,
	base_experience: int,
	height: int,
	weight: int,
	types: Pokemon_Type,
	abilities: string,
	moves: string ,
	stats: Pokemon_Stats
}

main :: proc() {

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

	data := clean_data()
	defer delete(data)
	process_data(data)
}

process_data :: proc(data: [dynamic]Pokemon) {
	layer1_weights := create_matrix(5,6)
	defer {
		for &row in layer1_weights {
			delete(row)
		}
		delete(layer1_weights)
	}
	for r in 0..<len(layer1_weights) {
		for c in 0..<len(layer1_weights[r]) {
			layer1_weights[r][c] = rand.float64_range(0.,5.)
		}
	}

	output_layer_weights := create_matrix(len(Pokemon_Type), 5)
	defer {
		for &row in output_layer_weights {
			delete(row)
		}
		delete(output_layer_weights)
	}
	for r in 0..<len(output_layer_weights ) {
		for c in 0..<len(output_layer_weights[r]) {
			output_layer_weights[r][c] = rand.float64_range(0.,5.)
		}
	}
	//find max and min
	bias:f64 = rand.float64_range(0.,10.)
	
	//Calculate layers

	for &input, index in data {
		input_data := []f64{
			input.stats.hp,
			input.stats.attack,
			input.stats.defense,
			input.stats.special_attack,
			input.stats.special_defense,
			input.stats.speed,
		}
		input_data = normalize_vector(input_data)
		layer1_neurons := dot(input_data, layer1_weights)
		defer delete(layer1_neurons)
		for &n in layer1_neurons {
			/* n += bias */
			n = relu(n)
		}
		layer1_neurons = normalize_vector(layer1_neurons[:])

		output_layer := dot(layer1_neurons, output_layer_weights)
		defer delete(output_layer )
		for &n in output_layer {
			/* n += bias */
		}
		output_layer = normalize_vector(output_layer[:])
		o := soft_max(output_layer[:])

		//Make decision
		sum:f64 = 0.
		chosen_output := -1
		curr_max := -1.
		for v, i in o {
			if v > curr_max {
				curr_max = v
				chosen_output = i
			}
			sum += v
		}

		//Calc loss
		fmt.println("OUTPUT VALUES:")
		fmt.println(o)
		expected_o := input.types
		if chosen_output > -1 {
			loss := cross_entropy_loss(o[expected_o])
			accuracy := 1 - math.abs(cross_entropy_loss(o[chosen_output]))

			fmt.println("STATS!!!")
			fmt.println("CHOSEN: ",cast(Pokemon_Type) chosen_output)
			fmt.println("EXPTED: ", input.types)
			fmt.println("LOSS: ", loss)
			fmt.println("ACCURACY: ", accuracy)

			//Back Prop

			//Update weights:
			//new_weight := old_weight - alpha * (Z * delta) WHERE alpha is your learning rate; Z is the loss of the current neuron and delta is the loss of the previous linked neuron

			//Find loss at each neuron weight * loss at output * partial derivative of activation function for that neuron
			/////UPDATE OUTPUT LAYER ///////

			//NOTE: This might be completely wrong and we onl;y care about out chosen output
			/* output_loss := make_slice([]f64, len(o))
			defer delete(output_loss)
			output_weights_to_update_t := transpose(output_layer_weights) 
			for row, r in output_weights_to_update_t {
				for col, c in output_weights_to_update_t[r] {
					weight := output_weights_to_update_t[r][c]
					d_a := cross_entropy_loss(output_layer[c])
					neuron_loss := weight * loss * d_a
					output_loss[c] = neuron_loss
				}
			} */


			/* fmt.println(output_layer_weights[chosen_output]) */
			alpha := 0.1
			/* output_layer_weights_t := transpose(output_layer_weights)  */
			for &row, r in output_layer_weights[chosen_output] {
				new_weight := row - alpha * (loss)
				row = new_weight
			}
			/* output_layer_weights = transpose(output_layer_weights_t) */

			//UPDATE LAYER 1 //////////

			//NOTE: Do I also only update the weights for the chosen output stuff?
			layer1_neuron_loss := make_slice([]f64, len(layer1_neurons))
			defer delete(layer1_neuron_loss)
			weights_to_update_t := transpose(layer1_weights) 
			for row, r in weights_to_update_t {
				for col, c in weights_to_update_t[r] {
					weight := weights_to_update_t[r][c]
					d_a := d_relu(layer1_neurons[c])
					neuron_loss := weight * loss * d_a
					layer1_neuron_loss[c] = neuron_loss
				}
			}

			for row, r in weights_to_update_t {
				for &col, c in weights_to_update_t[r] {
					new_weight := col - alpha * (layer1_neuron_loss[c] * loss)
					col = new_weight
				}
			}
			layer1_weights := transpose(weights_to_update_t)
			fmt.println()
		}
		fmt.println("\t======ITERATION: ", index)
	}
	fmt.println()
	print_matrix(layer1_weights)
	fmt.println()
	print_matrix(output_layer_weights)
}

weighted_sum :: proc(input, weights: []f64, bias: f64) -> f64 {
	assert(len(input) == len(weights))
	sum:f64 = 0.
	for i in 0..<len(input) {
		sum += input[i] * weights[i]
	}
	return sum + bias
}

relu :: proc(z: f64) -> f64 {
	return max(0., z)
}

d_relu :: proc(z: f64) -> f64 {
	return z >= 0. ? 1. : 0.
}

soft_max :: proc(x: []f64) -> [len(Pokemon_Type)]f64{
	assert(len(x) == len(Pokemon_Type))
	//For a given vector calculate:
	//sum of math.exp of each value
	//Then for each value it is exp(i)/ sum of exp(i)
	ret:[len(Pokemon_Type)]f64
	exp_sum:f64 = 0.
	for val in x {
		exp_sum += math.exp(val)
	}
	for val, index in x {
		ret[index] = math.exp(val)/exp_sum
	}
	return ret
}

calc_loss :: proc() {
	//
}

//Normally uses one-hot encoding to get the loss
//sum from 1 - number of classifications of y*log(y-hat) where y is the one-hot encoded entry and y-hat is the chosen class from prediction
cross_entropy_loss :: proc(predicted_val: f64) -> f64 {
	return -(math.log10(predicted_val))
}

dot :: proc(v: []f64, m:[][]f64) -> []f64 {
	assert(len(v) == len(m[0]))
	output := make_slice([]f64, len(m))
	
	for r in 0..<len(m) {
		curr_sum := 0.
		for c in 0..<len(m[r]) {
			curr_sum += v[c] * m[r][c]
		}
		output[r] = curr_sum
	}

	return output
}

normalize_vector :: proc(vector: []f64) -> []f64 {
	vector := vector
	min := 99999.
	max := -99999.
	for val in vector {
		if val < min {
			min = val
		}

		if val > max {
			max = val
		}
	}

	denom := max - min
	for &val in vector {
		numerator := val - min
		range := 1. - -1.
		val = (numerator/denom) * range + 1
	}
	return vector
}

create_matrix :: proc(rows, cols: int) -> [][]f64 {
	m := make_slice([][]f64 , rows)
	for &row in m {
		row = make_slice([]f64, cols)
	}
	return m 
}

transpose :: proc(input_matrix: [][]f64) -> [][]f64 {
	input_matrix_t := make_slice([][]f64, len(input_matrix[0]))
	for &row in input_matrix_t {
		row = make_slice([]f64, len(input_matrix))
	}

	for c in 0..<len(input_matrix[0]) {
		for r in 0..<len(input_matrix) {
			input_matrix_t[c][r] = input_matrix[r][c]
		}
	}
	return input_matrix_t
}

find_max :: proc(vector: []f64) -> (int, f64) {
	curr_i := -1
	curr_max := -99999999.
	for n, i in vector {
		if n > curr_max {
			curr_max = n
			curr_i = i
		}
	}
	return curr_i, curr_max
}

temp :: proc() {
	x := 500.
	z := math.exp_f64(x)
	fmt.println(z)
}

print_matrix :: proc(m: [][]f64) {
	for row in m{
		fmt.printfln("%v", row)
	}
}

clean_data :: proc() -> [dynamic]Pokemon {
	data, ok := os.read_entire_file_from_filename("pokemon_data.csv")
	defer delete(data)
	s := string(data)

	line_count := 0
	pokemon_data := make_dynamic_array([dynamic]Pokemon)
	for line in strings.split_lines_iterator(&s) {
		if line_count == 0 {
			line_count += 1
			continue
		}
		first_half, second_half := fetch_stats(line)
		first_half_arr := strings.split(first_half, ",")
		defer delete(first_half_arr)
		second_half_arr := strings.split(second_half, ",")
		defer delete(second_half_arr)
		curr_data:Pokemon
		curr_data.name= first_half_arr[1]
		curr_data.stats = convert_to_stats(second_half_arr)

		s_remove, ok := strings.remove(first_half_arr[5], "\"", 1)
		if ok {
			defer delete(s_remove)
			curr_data.types = convert_to_type(s_remove)
		} else {
			curr_data.types = convert_to_type(first_half_arr[5])
		}

		line_count += 1
		append(&pokemon_data, curr_data)
		/* if line_count >= 50 {
			break
		} */
	}
	return pokemon_data
}

fetch_stats :: proc(s: string) -> (first_half: string, second_half: string) {
	i := strings.index(s, "hp=")
	ok: bool
	first_half, ok = strings.substring(s, 0, i)
	second_half, ok = strings.substring(s, i, len(s)-1)
	return
}

convert_to_type :: proc(s: string) -> Pokemon_Type {
	fmt.println(s)
	data := strings.split(s, ",")
	defer delete(data)
	p_type: Pokemon_Type
	switch data[0] {
	case "fire": p_type = .FIRE
	case "grass": p_type = .GRASS
	case "water": p_type = .WATER
	case "flying": p_type = .FLYING
	case "ground": p_type = .GROUND
	case "rock": p_type = .ROCK
	case "poison": p_type = .POISON
	case "steel": p_type = .STEEL
	case "electric": p_type = .ELECTRIC
	case "fighting": p_type = .FIGHTING
	case "fairy": p_type = .FAIRY
	case "normal": p_type = .NORMAL
	case "bug": p_type = .BUG
	case "psychic": p_type = .PSYCHIC
	case "dark": p_type = .DARK
	case "ghost": p_type = .GHOST
	}
	return p_type
}

convert_to_stats :: proc(s: []string) -> Pokemon_Stats {
	stats: Pokemon_Stats
	hp_substring, hp_ok := strings.substring(s[0], strings.index(s[0], "=") + 1, len(s[0]))
	stats.hp = strconv.atof(hp_substring)

	attack_substring, attack_ok := strings.substring(s[1], strings.index(s[1], "=") + 1, len(s[1]))
	stats.attack = strconv.atof(attack_substring)

	defense_substring, defense_ok := strings.substring(s[2], strings.index(s[2], "=") + 1, len(s[2]))
	stats.defense = strconv.atof(defense_substring)

	special_attack_substring, special_attack_ok := strings.substring(s[3], strings.index(s[3], "=") + 1, len(s[3]))
	stats.special_attack = strconv.atof(special_attack_substring)

	special_defense_substring, special_defense_ok := strings.substring(s[4], strings.index(s[4], "=") + 1, len(s[4]))
	stats.special_defense = strconv.atof(special_defense_substring)

	speed_substring, speed_ok := strings.substring(s[5], strings.index(s[5], "=") + 1, len(s[5]))
	stats.speed = strconv.atof(speed_substring)

	return stats
}
