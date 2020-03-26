module SKPreprocessors

export SKPreprocessor,transform!,fit!

using TSML
import TSML.fit! # to overload
import TSML.transform! # to overload

using PyCall

function __init__()
  global DEC=pyimport_conda("sklearn.decomposition","scikit-learn") 
  global FS=pyimport_conda("sklearn.feature_selection","scikit-learn")
  global IMP=pyimport_conda("sklearn.impute","scikit-learn")
  global PREP=pyimport_conda("sklearn.preprocessing","scikit-learn")

  # Available scikit-learn learners.
  global preprocessor_dict = Dict(
     "DictionaryLearning" => DEC.DictionaryLearning,
     "FactorAnalysis" => DEC.FactorAnalysis,
     "FastICA" => DEC.FastICA,
     "IncrementalPCA" => DEC.IncrementalPCA,
     "KernelPCA" => DEC.KernelPCA,
     "LatentDirichletAllocation" => DEC.LatentDirichletAllocation,
     "MiniBatchDictionaryLearning" => DEC.MiniBatchDictionaryLearning,
     "MiniBatchSparsePCA" => DEC.MiniBatchSparsePCA,
     "NMF" => DEC.NMF,
     "PCA" => DEC.PCA, 
     "SparsePCA" => DEC.SparsePCA,
     "SparseCoder" => DEC.SparseCoder,
     "TruncatedSVD" => DEC.TruncatedSVD,
     "dict_learning" => DEC.dict_learning,
     "dict_learning_online" => DEC.dict_learning_online,
     "fastica" => DEC.fastica,
     "non_negative_factorization" => DEC.non_negative_factorization,
     "sparse_encode" => DEC.sparse_encode,
     "GenericUnivariateSelect" => FS.GenericUnivariateSelect,
     "SelectPercentile" => FS.SelectPercentile,
     "SelectKBest" => FS.SelectKBest,
     "SelectFpr" => FS.SelectFpr,
     "SelectFdr" => FS.SelectFdr,
     "SelectFromModel"  => FS.SelectFromModel,
     "SelectFwe" => FS.SelectFwe,
     "RFE" => FS.RFE,
     "RFECV" => FS.RFECV,
     "VarianceThreshold"  => FS.VarianceThreshold,
     "chi2" => FS.chi2,
     "f_classif"  => FS.f_classif,
     "f_regression" => FS.f_regression,
     "mutual_info_classif" => FS.mutual_info_classif,
     "mutual_info_regression" => FS.mutual_info_regression,
     "SimpleImputer" => IMP.SimpleImputer,
     #"IterativeImputer" => IMP.IterativeImputer,
     #"KNNImputer" => IMP.KNNImputer,
     "MissingIndicator" => IMP.MissingIndicator,
     "Binarizer" => PREP.Binarizer,
     "FunctionTransformer" => PREP.FunctionTransformer,
     "KBinsDiscretizer" => PREP.KBinsDiscretizer,
     "KernelCenterer" => PREP.KernelCenterer,
     "LabelBinarizer" => PREP.LabelBinarizer,
     "LabelEncoder" => PREP.LabelEncoder,
     "MultiLabelBinarizer" => PREP.MultiLabelBinarizer,
     "MaxAbsScaler" => PREP.MaxAbsScaler,
     "MinMaxScaler" => PREP.MinMaxScaler,
     "Normalizer" => PREP.Normalizer,
     "OneHotEncoder" => PREP.OneHotEncoder,
     "OrdinalEncoder" => PREP.OrdinalEncoder,
     "PolynomialFeatures" => PREP.PolynomialFeatures,
     "PowerTransformer" => PREP.PowerTransformer,
     "QuantileTransformer" => PREP.QuantileTransformer,
     "RobustScaler" => PREP.RobustScaler,
     "StandardScaler" => PREP.StandardScaler
     #"add_dummy_feature" => PREP.add_dummy_feature,
     #"binarize" => PREP.binarize,
     #"label_binarize" => PREP.label_binarize,
     #"maxabs_scale" => PREP.maxabs_scale,
     #"minmax_scale" => PREP.minmax_scale,
     #"normalize" => PREP.normalize,
     #"quantile_transform" => PREP.quantile_transform,
     #"robust_scale" => PREP.robust_scale,
     #"scale" => PREP.scale,
     #"power_transform" => PREP.power_transform
    )
end

mutable struct SKPreprocessor <: Transformer
    model
    args

    function SKPreprocessor(args=Dict())
        default_args=Dict(
           :preprocessor => "PCA",
           :impl_args => Dict()
        )
        new(nothing,mergedict(default_args,args))
    end
end

function fit!(skp::SKPreprocessor, x::DataFrame, y::Vector=[])
  features = x |> Array
  impl_args = copy(skp.args[:impl_args])
  preprocessor = skp.args[:preprocessor]
  py_preprocessor = preprocessor_dict[preprocessor]

  # Train model
  skp.model = py_preprocessor(;impl_args...)
  skp.model.fit(features)
end

function transform!(skp::SKPreprocessor, x::DataFrame)
  features = x |> Array
  #return collect(skl.model[:predict](x))
  return collect(skp.model.transform(features)) |> DataFrame
end

function skprun()

    iris=getiris()
    features=iris[:,1:4] |> Matrix
    labels=iris[:,5] |> Vector

    pca = SKPreprocessor(Dict(:preprocessor=>"PCA",:impl_args=>Dict(:n_components=>3)))
    fit!(pca,features)
    @assert transform!(pca,features) |> x->size(x,2) == 3

    svd = SKPreprocessor(Dict(:preprocessor=>"TruncatedSVD",:impl_args=>Dict(:n_components=>2)))
    fit!(svd,features)
    @assert transform!(svd,features) |> x->size(x,2) == 2

    ica = SKPreprocessor(Dict(:preprocessor=>"FastICA",:impl_args=>Dict(:n_components=>2)))
    fit!(ica,features)
    @assert transform!(ica,features) |> x->size(x,2) == 2


    stdsc = SKPreprocessor(Dict(:preprocessor=>"StandardScaler",:impl_args=>Dict()))
    fit!(stdsc,features)
    @assert abs(mean(transform!(stdsc,features))) < 0.00001

    minmax = SKPreprocessor(Dict(:preprocessor=>"MinMaxScaler",:impl_args=>Dict()))
    fit!(minmax,features)
    @assert mean(transform!(minmax,features)) ≈ 0.4486931104833648

    learner = VoteEnsemble()
    learner = StackEnsemble()
    learner = BestLearner()

    pipeline = Pipeline(Dict(
            :transformers => [stdsc,pca,learner]
    ))
    fit!(pipeline,features,labels)
    pred = transform!(pipeline,features)
    score(:accuracy,pred,labels)

end

end

