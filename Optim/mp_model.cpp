#include "../pch.hpp"

#include "mp_model.hpp"

#include "../Compute/gradients.hpp"

tc::optim::MPModel::MPModel(MPEvalDiffHessFunc func)
	: m_Func(func)
{
}

