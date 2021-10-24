export MNIST

"""
The MNIST database of handwritten digits
- Authors: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- Website: http://yann.lecun.com/exdb/mnist/
MNIST is a classic image-classification dataset that is often
used in small-scale machine learning experiments. It contains
70,000 images of handwritten digits. Each observation is a 28x28
pixel gray-scale image that depicts a handwritten version of 1 of
the 10 possible digits (0-9).
# Interface
- [`MNIST.traintensor`](@ref), [`MNIST.trainlabels`](@ref), [`MNIST.traindata`](@ref)
- [`MNIST.testtensor`](@ref), [`MNIST.testlabels`](@ref), [`MNIST.testdata`](@ref)
# Utilities
- [`MNIST.download`](@ref)
- [`MNIST.convert2image`](@ref)
"""
module MNIST
    using DataDeps
    using ColorTypes
    using FixedPointNumbers

    bytes_to_type(::Type{UInt8}, A::Array{UInt8}) = A
    bytes_to_type(::Type{N0f8}, A::Array{UInt8}) = reinterpret(N0f8, A)
    bytes_to_type(::Type{T}, A::Array{UInt8}) where T<:Integer = convert(Array{T}, A)
    bytes_to_type(::Type{T}, A::Array{UInt8}) where T<:AbstractFloat = A ./ T(255)
    bytes_to_type(::Type{T}, A::Array{UInt8}) where T<:Number  = convert(Array{T}, reinterpret(N0f8, A))

    function with_accept(f, manual_overwrite)
        auto_accept = if manual_overwrite == nothing
            get(ENV, "DATADEPS_ALWAYS_ACCEPT", false)
        else
            manual_overwrite
        end
        withenv(f, "DATADEPS_ALWAYS_ACCEPT" => string(auto_accept))
    end

    function datadir(depname, dir = nothing; i_accept_the_terms_of_use = nothing)
        with_accept(i_accept_the_terms_of_use) do
            if dir == nothing
                # use DataDeps defaults
                @datadep_str depname
            else
                # use user-provided dir
                if isdir(dir)
                    dir
                else
                    DataDeps.env_bool("DATADEPS_DISABLE_DOWNLOAD") && error("DATADEPS_DISABLE_DOWNLOAD enviroment variable set. Can not trigger download.")
                    DataDeps.download(DataDeps.registry[depname], dir)
                    dir
                end
            end
        end::String
    end

    function datafile(depname, filename, dir = nothing; recurse = true, kw...)
        path = joinpath(datadir(depname, dir; kw...), filename)
        if !isfile(path)
            @warn "The file \"$path\" does not exist, even though the dataset-specific folder does. This is an unusual situation that may have been caused by a manual creation of an empty folder, or manual deletion of the given file \"$filename\"."
            if dir === nothing
                @info "Retriggering DataDeps.jl for \"$depname\""
                download_dep(depname; kw...)
            else
                @info "Retriggering DataDeps.jl for \"$depname\" to \"$dir\"."
                download_dep(depname, dir; kw...)
            end
            if recurse
                datafile(depname, filename, dir; recurse = false, kw...)
            else
                error("The file \"$path\" still does not exist. One possible explaination could be a spelling error in the name of the requested file.")
            end
        else
            path
        end::String
    end

    function download_dep(depname, dir = DataDeps.determine_save_path(depname); kw...)
        DataDeps.download(DataDeps.registry[depname], dir; kw...)
    end

    function download_docstring(modname, depname)
        """
        The corresponding resource file(s) of the dataset is/are
        expected to be located in the specified directory `dir`. If
        `dir` is omitted the directories in
        `DataDeps.default_loadpath` will be searched for an existing
        `$(depname)` subfolder. In case no such subfolder is found,
        `dir` will default to `~/.julia/datadeps/$(depname)`. In the
        case that `dir` does not yet exist, a download prompt will be
        triggered. You can also use `$(modname).download([dir])`
        explicitly for pre-downloading (or re-downloading) the
        dataset. Please take a look at the documentation of the
        package DataDeps.jl for more detail and configuration
        options.
        """
    end

    function _colorview(::Type{T}, array::AbstractArray{<:Number}) where T <: Color
        __images_supported__ ||
            error("Converting to image requires `using ImageCore`")
        ImageCore.colorview(T, array)
    end

    export

        traintensor,
        testtensor,

        trainlabels,
        testlabels,

        traindata,
        testdata,

        convert2image,

        download

    @deprecate convert2features reshape

    const DEPNAME = "MNIST"
    const TRAINIMAGES = "train-images-idx3-ubyte.gz"
    const TRAINLABELS = "train-labels-idx1-ubyte.gz"
    const TESTIMAGES  = "t10k-images-idx3-ubyte.gz"
    const TESTLABELS  = "t10k-labels-idx1-ubyte.gz"

    """
        download([dir]; [i_accept_the_terms_of_use])
    Trigger the (interactive) download of the full dataset into
    "`dir`". If no `dir` is provided the dataset will be
    downloaded into "~/.julia/datadeps/$DEPNAME".
    This function will display an interactive dialog unless
    either the keyword parameter `i_accept_the_terms_of_use` or
    the environment variable `DATADEPS_ALWAYS_ACCEPT` is set to
    `true`. Note that using the data responsibly and respecting
    copyright/terms-of-use remains your responsibility.
    """
    download(args...; kw...) = download_dep(DEPNAME, args...; kw...)

    include(joinpath("Reader","Reader.jl"))
    include("interface.jl")
    include("utils.jl")

    function __init__()
        register(DataDep(
            DEPNAME,
            """
            Dataset: THE MNIST DATABASE of handwritten digits
            Authors: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
            Website: http://yann.lecun.com/exdb/mnist/
            [LeCun et al., 1998a]
                Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner.
                "Gradient-based learning applied to document recognition."
                Proceedings of the IEEE, 86(11):2278-2324, November 1998
            The files are available for download at the offical
            website linked above. Note that using the data
            responsibly and respecting copyright remains your
            responsibility. The authors of MNIST aren't really
            explicit about any terms of use, so please read the
            website to make sure you want to download the
            dataset.
            """,
            "https://ossci-datasets.s3.amazonaws.com/mnist/" .* [TRAINIMAGES, TRAINLABELS, TESTIMAGES, TESTLABELS],
            "0bb1d5775d852fc5bb32c76ca15a7eb4e9a3b1514a2493f7edfcf49b639d7975",
        ))
    end
end
