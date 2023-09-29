package main

import (
	"fmt"

	"github.com/ranon-rat/rnnTest/brain"
)

func main() {
	nn := brain.NewNeuralNetwork([]int{len(letters), 32, 32, 32, 32, 32, 32, len(letters)}, []string{"tanh", "tanh", "tanh", "tanh", "tanh", "tanh", "tanh"}, "sex")
	//nn = brain.OpenModel("its-gay.json")
	expected := "hello world"
	for k := 0; k < 10000; k++ {
		var bd [][][]float32
		var wd [][][][]float32
		var b [][]float32 = nil

		input := make([]float32, len(letters))
		input[la[expected[0]]] = 1

		for i := 1; i < len(expected); i++ {

			layers, bef := nn.FeedFoward(input, b)

			exp := make([]float32, len(letters))
			exp[la[expected[i]]] = 1

			w, bi := nn.BackPropagation(layers, b, exp)
			bd = append(bd, bi)
			wd = append(wd, w)
			input = layers[len(layers)-1]

			b = bef

		}
		for i := 0; i < len(bd); i++ {
			nn.UpdateWeightAndBias(float32(len(bd)), 0.001, wd[i], bd[i])
		}
		if k%100 == 0 {
			var bias [][]float32 = nil

			inp := make([]float32, len(letters))
			inp[la[expected[0]]] = 1

			fmt.Print(string(expected[0]))
			for i := 1; i < len(expected); i++ {
				out, bef := nn.Predict(inp, bias)
				inp = out
				//fmt.Println(out[len(bef)-1])
				fmt.Print(letters[brain.Argmax(out)])
				bias = bef
			}
			fmt.Println()
		}

	}
	nn.SaveModel("its-gay.json")
}

var letters = []string{
	"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", " ", ",",
}
var la = map[byte]int{
	'a': 0,
	'b': 1,
	'c': 2,
	'd': 3,
	'e': 4,
	'f': 5,
	'g': 6,
	'h': 7,
	'i': 8,
	'j': 9,
	'k': 10,
	'l': 11,
	'm': 12,
	'n': 13,
	'o': 14,
	'p': 15,
	'q': 16,
	'r': 17,
	's': 18,
	't': 19,
	'u': 20,
	'v': 21,
	'w': 22,
	'x': 23,
	'y': 24,
	'z': 25,
	' ': 26,
	',': 27,
}
