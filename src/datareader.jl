module DataReaders
using FileIO
using CSVFiles
using FeatherFiles
using Feather
using HDF5
using JLD
using Parquet
using DataFrames

export DataReader, fit!, transform!

using TSML
import TSML.fit! # to overload
import TSML.transform! # to overload

const FILEFMT = Dict("csv"=>Val(:csv),"feather"=>Val(:feather),"hdf5"=>Val(:hdf5),
          "h5"=>Val(:hdf5),"parquet"=>Val(:parquet),"jld"=>Val(:jld))

const DATEFMT = "d/m/yyyy HH:MM"

mutable struct DataReader <: Transformer
    model
    args

    function DataReader(args=Dict())
        default_args=Dict(
            :filename => "",
            :dateformat => DATEFMT,
            :impl_args => Dict()
        )
        new(nothing,mergedict(default_args,args))
    end
end

function fit!(dtr::DataReader,x::DataFrame=DataFrame(),y::Vector=[])
    fname = dtr.args[:filename]
    fmt = dtr.args[:dateformat]
    (fname != "" && isfile(fname) && fmt != "") || error("missing filename or date format: (",fname,"), (",fmt,")")
    dtr.model = dtr.args
end

function transform!(dtr::DataReader,x::DataFrame=DataFrame())
    fullname = dtr.args[:filename]
    fmt = dtr.args[:dateformat]
    (fullname != "" && isfile(fullname) && fmt != "") || error("missing filename or date format: (",fullname,"), (",fmt,")")
    fname = basename(fullname)
    fname != "" || error("filename is empty")
    fn,ext=split(fname,".")
    ext != "" || error("no filename extension format")
    ext in keys(FILEFMT)  || error("extension not recognized "*ext)
    # dispatch based on extension
    readfmt(FILEFMT[ext],fullname,fmt)
end

function readfmt(::T,fname::String,datefmt::String) where {T<:Union{Val{:csv},Val{:feather},Val{:parquet}}}
    df = load(fname) |> DataFrame
    ncol(df) == 2 || error("dataframe should have only two columns: Date,Value")
    rename!(df,names(df)[1]=>:Date,names(df)[2]=>:Value)
    if eltype(df.Date) <: DateTime
        return df
    else
        df.Date = DateTime.(df.Date,datefmt)
        return df
    end
end


function readfmt(atype::Union{Val{:hdf5},Val{:jld}},fname::String,datefmt::String)
    # R library(hdf5r); dat=H5File$new(fname);
    # dat[["dateval/date"]];dat[["dateval/value"]]
    if atype == Val{:hdf5}
        fileopen = h5open
    else
        fileopen = jldopen
    end
    ldate = fileopen(fname, "r") do file
        Base.read(file, "dateval/date")
    end
    lvalue = fileopen(fname, "r") do file
        Base.read(file, "dateval/value")
    end
    DataFrame(date=DateTime.(ldate,datefmt),Value=lvalue)
end

end
