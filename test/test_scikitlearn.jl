module TestSKL

using Random
using TSML
using TSMLextra
using PyCall
using Test

const IRIS = getiris()
const X = IRIS[:,1:4] |> DataFrame
const Y = IRIS[:,5] |> Vector

const XX = IRIS[:,2:4] |> DataFrame
const YY = IRIS[:,1] |> Vector

const classifiers = [
    "LinearSVC","QDA","MLPClassifier","BernoulliNB",
    "RandomForestClassifier",
    "NearestCentroid","SVC","LinearSVC","NuSVC","MLPClassifier",
    "SGDClassifier","KNeighborsClassifier",
    "DecisionTreeClassifier",
    "PassiveAggressiveClassifier","RidgeClassifier",
    "ExtraTreesClassifier","GradientBoostingClassifier",
    "BaggingClassifier","AdaBoostClassifier","GaussianNB","MultinomialNB",
    "ComplementNB","BernoulliNB"
    #"RidgeClassifierCV","LDA",
    #"GaussianProcessClassifier",
 ]

const regressors = [
    "SVR",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "Lars",
    "LassoLars",
    "OrthogonalMatchingPursuit",
    "SGDRegressor",
    "PassiveAggressiveRegressor",
    "KNeighborsRegressor",
    "RadiusNeighborsRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "MLPRegressor",
    "AdaBoostRegressor"
    #"RidgeCV",
    #"BayesianRidge",
    #"ARDRegression",
    #"KernelRidge",
    #"GaussianProcessRegressor",
    #"IsotonicRegression",
]
    	

function fit_test(learner::String,in::DataFrame,out::Vector)
	_learner=SKLearner(Dict(:learner=>learner))
	fit!(_learner,in,out)
	@test _learner.model != nothing
	return _learner
end

function fit_transform_reg(model::Learner,in::DataFrame,out::Vector)
    @test sum((transform!(model,in) .- out).^2)/length(out) < 10.0
end

@testset "scikit classifiers" begin
    Random.seed!(1)
    for cl in classifiers
	println(cl)
	fit_test(cl,X,Y)
    end
end

@testset "scikit regressors" begin
    Random.seed!(1)
    for rg in regressors
	println(rg)
	model=fit_test(rg,XX,YY)
	fit_transform_reg(model,XX,YY)
    end
end

end
