#include "optim.hpp"
#include "optim.hpp"

#include "../Compute/gradients.hpp"


tc::optim::OptimizerSettings::OptimizerSettings()
	: startDevice(torch::Device("cpu"))
{

}
