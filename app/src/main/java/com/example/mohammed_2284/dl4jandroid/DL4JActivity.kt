package com.example.mohammed_2284.dl4jandroid

import android.support.v7.app.AppCompatActivity
import android.os.Bundle
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import android.os.AsyncTask
import android.text.TextWatcher
import kotlinx.android.synthetic.main.dl4j_activity.*
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import android.text.Editable
import android.text.TextUtils
import android.view.View
import android.widget.Toast
import org.apache.commons.math3.util.IterationEvent
import org.apache.commons.math3.util.IterationListener
import org.deeplearning4j.nn.api.Model
import org.nd4j.linalg.activations.Activation


//  INPUTS      EXPECTED OUTPUTS
//  ------      ----------------
//  0,0         0
//  0,1         1
//  1,0         1
//  1,1         0

class DL4JActivity : AppCompatActivity() {

    val learningRate = .1
    val epochs = 60000
    val displayStep = 1000

    val NUM_SAMPLES = 4

    lateinit var myNetwork: MultiLayerNetwork

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.dl4j_activity)
        initViews()
        AsyncTask.execute { justANNDemo() }
    }

    //  initViews
    fun initViews() {
        val actualInput = Nd4j.zeros(1, 2)
        predictButton.setOnClickListener {
            if (TextUtils.isEmpty(input1.text)
                    || TextUtils.isEmpty(input2.text)) {
                Toast.makeText(applicationContext, "empty values not allowed", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            actualInput.putScalar(intArrayOf(0, 0), 0)
            actualInput.putScalar(intArrayOf(0, 1), 0)
            actualInput.putScalar(intArrayOf(0, 0), input1.text.toString().toInt())
            actualInput.putScalar(intArrayOf(0, 1), input2.text.toString().toInt())
            //  Generate output
            val actualOutput = myNetwork.output(actualInput)
            resultTxtView.text = actualOutput.toString()
            println("---test actualOutput.toString() :: " + actualOutput.toString())
        }
        input1.addTextChangedListener(object : TextWatcher {
            override fun onTextChanged(s: CharSequence, start: Int, before: Int, count: Int) {
                if (s.isNotEmpty()) {
                    if (s.toString().toInt() != 1 && s.toString().toInt() != 0) {
                        input1.setText("")
                    }
                }
            }

            override fun beforeTextChanged(s: CharSequence, start: Int, count: Int, after: Int) {}

            override fun afterTextChanged(s: Editable) {
            }
        })

        input2.addTextChangedListener(object : TextWatcher {
            override fun onTextChanged(s: CharSequence, start: Int, before: Int, count: Int) {
                if (s.isNotEmpty()) {
                    if (s.toString().toInt() != 1 && s.toString().toInt() != 0) {
                        input2.setText("")
                    }
                }
            }

            override fun beforeTextChanged(s: CharSequence, start: Int, count: Int, after: Int) {}

            override fun afterTextChanged(s: Editable) {
            }
        })
    }

    //  just a demo
    fun justANNDemo() {
        val inputLayer = DenseLayer.Builder()
                .nIn(2)
                .nOut(3)
                .name("Input")
                .build()

        val hiddenLayer = DenseLayer.Builder()
                .nIn(3)
                .nOut(2)
                .name("Hidden")
                .build()

        val outputLayer = OutputLayer.Builder()
                .nIn(2)
                .nOut(2)
                .name("Output")
                .activation(Activation.SOFTMAX)
                .build()

        val nncBuilder = NeuralNetConfiguration.Builder()
        nncBuilder.iterations(epochs)
        nncBuilder.learningRate(learningRate)

        val listBuilder = nncBuilder.list()
        listBuilder.layer(0, inputLayer)
        listBuilder.layer(1, hiddenLayer)
        listBuilder.layer(2, outputLayer)
        listBuilder.backprop(true)

//        val myNetwork = MultiLayerNetwork(listBuilder.build())

        myNetwork = MultiLayerNetwork(listBuilder.build())
        //  nn init
        myNetwork.init()
        myNetwork.addListeners(object : IterationListener, org.deeplearning4j.optimize.api.IterationListener {

            override fun iterationDone(model: Model?, iteration: Int) {
                runOnUiThread {
                    if ((iteration + 1) == epochs) {
                        pBar.visibility = View.GONE
                        predictButton.isEnabled = true
                    }

                    if ((iteration + 1) % displayStep == 0) {
                        println("---test completed step :: " + (iteration + 1))
                    }
                }
            }

            override fun invoked(): Boolean {
                return true
            }

            override fun invoke() {}

            override fun terminationPerformed(e: IterationEvent?) {}

            override fun initializationPerformed(e: IterationEvent?) {}

            override fun iterationStarted(e: IterationEvent?) {}

            override fun iterationPerformed(e: IterationEvent?) {}
        })

        //  data
        val trainingInputs = Nd4j.zeros(NUM_SAMPLES, inputLayer.nIn)
        val trainingOutputs = Nd4j.zeros(NUM_SAMPLES, outputLayer.nOut)

        // If 0,0 show 0
        trainingInputs.putScalar(intArrayOf(0, 0), 0)
        trainingInputs.putScalar(intArrayOf(0, 1), 0)

        trainingOutputs.putScalar(intArrayOf(0, 0), 0)

        // If 0,1 show 1
        trainingInputs.putScalar(intArrayOf(1, 0), 0)
        trainingInputs.putScalar(intArrayOf(1, 1), 1)

        trainingOutputs.putScalar(intArrayOf(1, 0), 1)

        // If 1,0 show 1
        trainingInputs.putScalar(intArrayOf(2, 0), 1)
        trainingInputs.putScalar(intArrayOf(2, 1), 0)

        trainingOutputs.putScalar(intArrayOf(2, 0), 1)

        // If 1,1 show 0
        trainingInputs.putScalar(intArrayOf(3, 0), 1)
        trainingInputs.putScalar(intArrayOf(3, 1), 1)

        trainingOutputs.putScalar(intArrayOf(3, 0), 0)

        val myData = DataSet(trainingInputs, trainingOutputs)

        //  training
        myNetwork.fit(myData)

//        //  testing
//        // Create input
        val actualInput = Nd4j.zeros(1, 2)
//        actualInput.putScalar(intArrayOf(0, 0), 1)
//        actualInput.putScalar(intArrayOf(0, 1), 1)
        actualInput.putScalar(intArrayOf(0, 0), 0)
        actualInput.putScalar(intArrayOf(0, 1), 0)

        // Generate output
        val actualOutput = myNetwork.output(actualInput)
        println("network output is :: " + actualOutput)
        println("network output toString() is :: " + actualOutput.toString())
    }
}