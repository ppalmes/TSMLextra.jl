module TestSKL

using Random
using TSML
using TSMLextra
using PyCall
using Test

const IRIS = getiris()
const X = IRIS[:,1:4] |> DataFrame
const Y = IRIS[:,5] |> Vector

const XX = IRIS[:,1:3] |> DataFrame
const YY = IRIS[:,4] |> Vector

const classifiers = [
    "LinearSVC","QDA","MLPClassifier","BernoulliNB",
    "RandomForestClassifier","LDA",
    "NearestCentroid","SVC","LinearSVC","NuSVC","MLPClassifier",
    "RidgeClassifierCV","SGDClassifier","KNeighborsClassifier",
    "GaussianProcessClassifier","DecisionTreeClassifier",
    "PassiveAggressiveClassifier","RidgeClassifier",
    "ExtraTreesClassifier","GradientBoostingClassifier",
    "BaggingClassifier","AdaBoostClassifier","GaussianNB","MultinomialNB",
    "ComplementNB","BernoulliNB"
 ]

const regressors = [
    "SVR",
    "Ridge",
    "RidgeCV",
    "Lasso",
    "ElasticNet",
    "Lars",
    "LassoLars",
    "OrthogonalMatchingPursuit",
    "BayesianRidge",
    "ARDRegression",
    "SGDRegressor",
    "PassiveAggressiveRegressor",
    "KernelRidge",
    "KNeighborsRegressor",
    "RadiusNeighborsRegressor",
    "GaussianProcessRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "MLPRegressor",
    "AdaBoostRegressor"
#    "IsotonicRegression"
]
    	

function fit_test(learner::String,in::DataFrame,out::Vector)
	_learner=SKLearner(Dict(:learner=>learner))
	fit!(_learner,in,out)
	@test _learner.model != nothing
	return _learner
end

function fit_transform_reg(model::TSLearner,in::DataFrame,out::Vector)
    @test sum((transform!(model,in) .- out).^2)/length(out) < 2.0
end

@testset "scikit classifiers" begin
    Random.seed!(123)
    for cl in classifiers
	fit_test(cl,X,Y)
    end
end

@testset "scikit regressors" begin
    Random.seed!(123)
    for rg in regressors
	model=fit_test(rg,XX,YY)
	fit_transform_reg(model,XX,YY)
    end
end

end
