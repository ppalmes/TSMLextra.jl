module TSMLextra

using Reexport

@reexport using TSML
export fit!, transform!

include("system.jl")
using .System

include("datareader.jl")
using .DataReaders
export DataReader

include("datawriter.jl")
using .DataWriters
export DataWriter

if LIB_CRT_AVAILABLE # from System module
    include("caret.jl")
    using .CaretLearners
    export CaretLearner
end

if LIB_SKL_AVAILABLE # from System module
    include("scikitlearn.jl")
    using .SKLearners
    export SKLearner
    include("preprocessor.jl")
    using .SKPreprocessors
    export SKPreprocessor
end

end # module
