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
		rcall(:library,package,"lib=Sys.getenv('R_LIBS_USER')")
	catch
		R"dir.create(path = Sys.getenv('R_LIBS_USER'), showWarnings = FALSE, recursive = TRUE)"
		R"install.packages($package,lib=Sys.getenv('R_LIBS_USER'),repos='https://cloud.r-project.org')"
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
