using PyCall: pyimport, pycall
using RCall, Conda

function installpypackage()
	try
		pyimport("sklearn")
	catch
		Conda.add("scikit-learn")
	end
end

function installrpackage(package::AbstractString)
	try
		rcall(:library,package)
	catch
		R"install.packages($package,repos='https://cloud.r-project.org')"
	end
end

function installrml()
	packages=["caret", "earth","mda","e1071","gam","randomForest","nnet","kernlab","grid","MASS","pls","xgboost"]
	for pk in packages
		installrpackage(pk)
	end
end

installrml()
installpypackage()
