using PyCall: pyimport, pycall
using RCall, Conda

function installpypackage(package::AbstractString)
	try
		pyimport(package)
	catch
		try
			Conda.add(package)
		catch
		 	println("Installation of "*package*" failed")	
		end
	end
end

function installrpackage(package::AbstractString)
	try
		rcall(:library,package)
	catch
		try
			R"install.packages($package,repos='https://cloud.r-project.org',type='binary')"
		catch
		 	println("Installation of "*package*" failed")	
		end
	end
end

function installrml()
	packages=["caret", "earth","mda","e1071","gam","randomForest","nnet","kernlab","grid","MASS","pls"]
	for pk in packages
		installrpackage(pk)
	end
end

installrml()
installpypackage("scikit-learn")
