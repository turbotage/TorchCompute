#include "optim.hpp"
#include "optim.hpp"

#include "../Compute/gradients.hpp"


optim::OptimizerSettings::OptimizerSettings()
	: startDevice(torch::Device("cpu"))
{

}
