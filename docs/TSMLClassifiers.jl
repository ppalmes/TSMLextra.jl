using TSML
using TSMLextra
using Distributed
using Base.Threads
nthreads()

nprocs()==1 && addprocs()
nworkers()


@everywhere function predict(learner,data,train_ind,test_ind)
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

function parallelmodel(learners::Dict,data::DataFrame;trials=5)
    models=collect(keys(learners))
    ctable=@distributed (vcat) for i=1:trials
        # Split into training and test sets
        Random.seed!(3i)
        (train_ind, test_ind) = holdout(size(data, 1), 0.20)
        acc=@distributed (vcat) for model in models
            res,_=predict(learners[model],data,train_ind,test_ind)
            println("trial ",i,", ",model," => ",round(res))
            [model res i]
        end
        acc
    end
    df = ctable |> DataFrame
    rename!(df,:x1=>:model,:x2=>:acc,:x3=>:trial)
    gp=by(df,:model) do x
       DataFrame(mean=mean(x.acc),std=std(x.acc),n=length(x.acc))
    end
    sort!(gp,:mean,rev=true)
    return gp
end


@everywhere using TSML
@everywhere using TSMLextra

# Caret ML
@everywhere caret_svmlinear = CaretLearner(Dict(:learner=>"svmLinear"))
@everywhere caret_treebag = CaretLearner(Dict(:learner=>"treebag"))
@everywhere caret_rpart = CaretLearner(Dict(:learner=>"rpart"))
@everywhere caret_rf = CaretLearner(Dict(:learner=>"rf"))

# ScikitLearn ML
@everywhere sk_ridge = SKLearner(Dict(:learner=>"RidgeClassifier"))
@everywhere sk_sgd = SKLearner(Dict(:learner=>"SGDClassifier"))
@everywhere sk_knn = SKLearner(Dict(:learner=>"KNeighborsClassifier"))
@everywhere sk_gb = SKLearner(Dict(:learner=>"GradientBoostingClassifier",:impl_args=>Dict(:n_estimators=>10)))
@everywhere sk_extratree = SKLearner(Dict(:learner=>"ExtraTreesClassifier",:impl_args=>Dict(:n_estimators=>10)))
@everywhere sk_rf = SKLearner(Dict(:learner=>"RandomForestClassifier",:impl_args=>Dict(:n_estimators=>10)))

# Julia ML
@everywhere jrf = RandomForest(Dict(:impl_args=>Dict(:num_trees=>300)))
@everywhere jpt = PrunedTree()
@everywhere jada = Adaboost()

# Julia Ensembles
@everywhere jvote_ens=VoteEnsemble(Dict(:learners=>[jrf,jpt,sk_gb,sk_extratree,sk_rf]))
@everywhere jstack_ada=StackEnsemble(Dict(:stacker=>Adaboost(),:learners=>[jrf,jpt,sk_gb,sk_extratree,sk_rf]))
@everywhere jstack_rf=StackEnsemble(Dict(:learners=>[jrf,jpt,sk_gb,sk_extratree,sk_rf]))
@everywhere jbest_ens=BestLearner(Dict(:learners=>[jrf,sk_gb,sk_rf]))
@everywhere jsuper_ens=VoteEnsemble(Dict(:learners=>[jvote_ens,jstack_ada,jstack_rf,jbest_ens,sk_rf,sk_gb]))

learners=Dict(
      :jvote_ens=>jvote_ens,:jstack_rf=>jstack_rf,:jbest_ens=>jbest_ens, :jstack_ada=>jstack_ada,
      :jrf => jrf,:jada=>jada,:jsuper_ens=>jsuper_ens,:crt_rpart=>caret_rpart,
      :crt_svmlinear=>caret_svmlinear,:crt_treebag=>caret_treebag,:crt_rf=>caret_rf,
      :skl_knn=>sk_knn,:skl_gb=>sk_gb,:skl_extratree=>sk_extratree,
      :sk_rf => sk_rf
);

# use iris dataset
using RCall
iris = R"iris"|> rcopy
first(iris,5)

df = parallelmodel(learners,iris;trials=3)

(train_ind, test_ind) = holdout(size(iris, 1), 0.20)
res,workflow=predict(learners[:jstack_rf],iris,train_ind,test_ind)
res

showtree(workflow)

function threadedmodel(learners::Dict,data::DataFrame;trials=5)
    models=collect(keys(learners))
    global ctable = DataFrame()
    mtx = ReentrantLock()
    @threads for i=1:trials
        # Split into training and test sets
        Random.seed!(3i)
        (train_ind, test_ind) = holdout(size(data, 1), 0.20)
        @threads for model in models
            res=predict(learners[model],data,train_ind,test_ind)
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

df = threadedmodel(learners,tsdata;trials=5)
