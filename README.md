# Capsules Network Implementation in PyTorch

Work done with Pau Rué (https://github.com/paurue).

Reproduction of the model from the paper “Dynamic Routing Between Capsules” (https://arxiv.org/abs/1710.09829). This model achieves the best performance in the CapsNet implementations available in GitHub that publishes their results (as of 31 March 2018). 

## Why another model?

After reviewing a number of implementations of the model, we found several bugs on them. Once these bugs were corrected, we still were not able to arrive close to the performance of the paper. The main reason for that is in the way the optimisation is performed, which is not fully explained in the original paper. The authors employ an exponential decay on the learning rate, with a decay rate of 0.96 which is updated every 2000 steps.

The model results are very sensitive to the correct selection of these parameters. We were able to arrive to 0.28% error rate, which is very close to the performance of the paper (0.25%). We believe that with further tuning a 0.25% error rate is attainable.

Other critical parts that needs to be correctly addressed to make the model work are:
- There should not be backpropagation through each of the steps of the routing mechanism, only on the last step (use `.detach` method).
- The individual capsules of the second layer (PrimaryCaps) should be built with the `.view` method in the correct way, and for that one needs to set the dimensions in the correct order (in our case we use `.permute`). Many implementations were reshaping the tensor incorrectly, and hence the resulting capsules were not transversal to the filters, but contained in the filters.

We hope this implementation may be helpful for other researchers.

## Other Implementations

- PyTorch:
  - [XifengGuo/CapsNet-Pytorch](https://github.com/XifengGuo/CapsNet-Pytorch)
  - [timomernick/pytorch-capsule](https://github.com/timomernick/pytorch-capsule)
  - [gram-ai/capsule-networks](https://github.com/gram-ai/capsule-networks)
  - [nishnik/CapsNet-PyTorch](https://github.com/nishnik/CapsNet-PyTorch.git)
  - [leftthomas/CapsNet](https://github.com/leftthomas/CapsNet)
  - [acburigo/CapsNet](https://github.com/acburigo/CapsNet)
  
- TensorFlow:
  - [naturomics/CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow.git)   
  - [InnerPeace-Wu/CapsNet-tensorflow](https://github.com/InnerPeace-Wu/CapsNet-tensorflow)   
  - [chrislybaer/capsules-tensorflow](https://github.com/chrislybaer/capsules-tensorflow)

- MXNet:
  - [AaronLeong/CapsNet_Mxnet](https://github.com/AaronLeong/CapsNet_Mxnet)
  
- Chainer:
  - [soskek/dynamic_routing_between_capsules](https://github.com/soskek/dynamic_routing_between_capsules)

- Matlab:
  - [yechengxi/LightCapsNet](https://github.com/yechengxi/LightCapsNet)

(Credit to XifengGuo for compiling the list)


