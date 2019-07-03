module TSMLextra

greet() = print("Hello World!")

include("system.jl")
using .System

include("multilearner.jl")
using .MultiLearners

include("datawriter.jl")
using .DataWriters

include("datareader.jl")
using .DataReaders

include("dataproc.jl")
using .DataProc

if LIB_CRT_AVAILABLE # from System module
    include("caret.jl")
    using .CaretLearners
end

if LIB_SKL_AVAILABLE # from System module
    include("scikitlearn.jl")
    using .SKLearners
end


end # module
