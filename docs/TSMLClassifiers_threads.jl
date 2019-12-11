using TSML
using TSMLextra
using DataFrames
using Base.Threads
nthreads()

function predict(learner,data,train_ind,test_ind)
    features = convert(Matrix,data[:, 1:(end-1)])
    labels = convert(Array,data[:, end])
    # Create pipeline
    pipeline = Pipeline(
       Dict(
         :transformers => [
           OneHotEncoder(), # Encodes nominal features into numeric
           Imputer(), # Imputes NA values
           StandardScaler(),
           learner # Predicts labels on instances
         ]
       )
    )
    # Train
    fit!(pipeline, features[train_ind, :], labels[train_ind]);
    # Predict
    predictions = transform!(pipeline, features[test_ind, :]);
    # Assess predictions
    result = score(:accuracy, labels[test_ind], predictions)
    return result,pipeline
end

function threadedmodel(learners::Dict,data::DataFrame;trials=5)
    models=collect(keys(learners))
    global ctable = DataFrame()
    mtx = ReentrantLock()
    @threads for i=1:trials
        # Split into training and test sets
        Random.seed!(3i)
        (train_ind, test_ind) = holdout(size(data, 1), 0.20)
        @threads for model in models
            res,_=predict(learners[model],data,train_ind,test_ind)
            println("thread ",threadid(),", ",model," => ",round(res))
            lock(mtx)
            global ctable = vcat(ctable,DataFrame(model=model,acc=res))
            unlock(mtx)
        end
    end
    df = ctable |> DataFrame
    gp=by(df,:model) do x
       DataFrame(mean=mean(x.acc),std=std(x.acc),n=nrow(x))
    end
    sort!(gp,:mean,rev=true)
    return gp
end

# Caret ML
caret_svmlinear = CaretLearner(Dict(:learner=>"svmLinear"))
caret_treebag = CaretLearner(Dict(:learner=>"treebag"))
caret_rpart = CaretLearner(Dict(:learner=>"rpart"))
caret_rf = CaretLearner(Dict(:learner=>"rf"))

# ScikitLearn ML
sk_ridge = SKLearner(Dict(:learner=>"RidgeClassifier"))
sk_sgd = SKLearner(Dict(:learner=>"SGDClassifier"))
sk_knn = SKLearner(Dict(:learner=>"KNeighborsClassifier"))
sk_gb = SKLearner(Dict(:learner=>"GradientBoostingClassifier",:impl_args=>Dict(:n_estimators=>10)))
sk_extratree = SKLearner(Dict(:learner=>"ExtraTreesClassifier",:impl_args=>Dict(:n_estimators=>10)))
sk_rf = SKLearner(Dict(:learner=>"RandomForestClassifier",:impl_args=>Dict(:n_estimators=>10)))

# Julia ML
jrf = RandomForest(Dict(:impl_args=>Dict(:num_trees=>300)))
jpt = PrunedTree()
jada = Adaboost()

# Julia Ensembles
jvote_ens=VoteEnsemble(Dict(:learners=>[jrf,jpt,sk_gb,sk_extratree,sk_rf]))
jstack_ada=StackEnsemble(Dict(:stacker=>Adaboost(),:learners=>[jrf,jpt,sk_gb,sk_extratree,sk_rf]))
jstack_rf=StackEnsemble(Dict(:learners=>[jrf,jpt,sk_gb,sk_extratree,sk_rf]))
jbest_ens=BestLearner(Dict(:learners=>[jrf,sk_gb,sk_rf]))
jsuper_ens=VoteEnsemble(Dict(:learners=>[jvote_ens,jstack_ada,jstack_rf,jbest_ens,sk_rf,sk_gb]))


jstack_rf=StackEnsemble(Dict(:learners=>[jrf,jpt,jada]))
jbest_ens=BestLearner(Dict(:learners=>[jrf,jpt,jada]))
jstack_ada=StackEnsemble(Dict(:stacker=>Adaboost(),:learners=>[jrf,jpt,jada]))
jvote_ens=VoteEnsemble(Dict(:learners=>[jrf,jpt,jada]))
jsuper_ens=VoteEnsemble(Dict(:learners=>[jvote_ens,jstack_ada,jstack_rf,jbest_ens]))

#learners=Dict(
#      :jvote_ens=>jvote_ens,:jstack_rf=>jstack_rf,:jbest_ens=>jbest_ens, :jstack_ada=>jstack_ada,
#      :jrf => jrf,:jada=>jada,:jsuper_ens=>jsuper_ens,:crt_rpart=>caret_rpart,
#      :skl_knn=>sk_knn,:skl_gb=>sk_gb,:skl_extratree=>sk_extratree,
#      :sk_rf => sk_rf
#);


learners=Dict(
      :jrf => jrf,:jada=>jada,:jpt=>jpt,
      :jvote_ens=>jvote_ens,:jstack_rf=>jstack_rf,:jbest_ens=>jbest_ens,:jstack_ada=>jstack_ada,
      :jsuper_ens=>jsuper_ens
);

# use iris dataset
using RCall
iris = R"iris"|> rcopy
first(iris,5)

df = threadedmodel(learners,iris;trials=10)

function extract_features_from_timeseries(datadir)
  println("*** Extracting features ***")
  mdata = getstats(datadir)
  mdata[!,:dtype] = mdata[!,:dtype] |> Array{String}
  return mdata[!,3:(end-1)]
end

datadir = "data/realdatatsclassification/training"
tsdata = extract_features_from_timeseries(datadir)
first(tsdata,5)

df = threadedmodel(learners,tsdata;trials=20)

(train_ind, test_ind) = holdout(size(tsdata, 1), 0.20)
res,workflow=predict(learners[:jbest_ens],tsdata,train_ind,test_ind)
res

showtree(workflow)
