package main

import (
	"fmt"
	"log"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	celsiusQ := []float64{-40, -10, 0, 8, 15, 22, 38}
	fahrenheitA := []float64{-40, 14, 32, 46, 59, 72, 100}

	celsiusTensor := tensor.New(tensor.WithShape(len(celsiusQ), 1), tensor.Of(tensor.Float64), tensor.WithBacking(celsiusQ))
	fahrenheitTensor := tensor.New(tensor.WithShape(len(fahrenheitA), 1), tensor.Of(tensor.Float64), tensor.WithBacking(fahrenheitA))

	g := gorgonia.NewGraph()
	w := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, 1), gorgonia.WithInit(gorgonia.GlorotU(1)))
	x := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(len(celsiusQ), 1), gorgonia.WithValue(celsiusTensor))
	y := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(x, w)), gorgonia.NewScalar(g, tensor.Float64, gorgonia.WithValue(0))))

	loss := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Square(gorgonia.Must(gorgonia.Sub(y, fahrenheitTensor))))))

	learnRate := 0.1
	grads, err := gorgonia.Gradient(loss, w)
	if err != nil {
		log.Fatal(err)
	}
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(w))
	grad := gorgonia.NewTensor(grads[0].(gorgonia.Node), gorgonia.BindDualValues(w))

	epochs := 500
	for epoch := 0; epoch < epochs; epoch++ {
		if err := vm.RunAll(); err != nil {
			log.Fatal(err)
		}

		w.Value().Data().SubInPlace(gorgonia.NewScalar(tensor.Float64, gorgonia.WithValue(learnRate)), grad)
		vm.Reset()
	}

	modelPath := "model_savedmodel"
	if err := os.Mkdir(modelPath, os.ModePerm); err != nil {
		log.Fatal(err)
	}
	gorgonia.SaveGraph(g, modelPath+"/graph")
	gorgonia.SaveValue(w.Value(), modelPath+"/weights")

	fmt.Println("Finished training the model and saved as model_savedmodel")
}
