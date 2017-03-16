# locally register NIfTI format to FileIO, pending https://github.com/JuliaIO/NIfTI.jl/issues/7
add_format(format"NIfTI", (), ".nii")
add_loader(format"NIfTI", :NIfTI)
add_saver(format"NIfTI", :NIfTI)

FileIO.load(f::File{format"NIfTI"}; kwargs...) = niread(filename(f), kwargs...)
FileIO.save(f::File{format"NIfTI"}, data) = niwrite(filename(f), data)

function readDIRLab(nii::NIVolume)
    spacings = nii.header.pixdim[2:4]
    imageDims = nii.header.dim[2:4]
    data = squeeze(nii.raw, (4,5,6))
    img = AxisArray(permutedims(data, [2,1,3]),
                    Axis{:A}(Ranges.linspace(1u"mm", imageDims[2]*spacings[2]u"mm", imageDims[2])),
                    Axis{:R}(Ranges.linspace(1u"mm", imageDims[1]*spacings[1]u"mm", imageDims[1])),
                    Axis{:S}(Ranges.linspace(1u"mm", imageDims[3]*spacings[3]u"mm", imageDims[3])))
end


function readDIRLab(path, refDict, mode="r")
    endswith(basename(path), ".img") || throw(ArgumentError("Wrong file format, expect *.img, got $path."))
    spacings = refDict["Voxels"]
    imageDims = refDict["Image Dims"]
    data = open(path, mode) do io
        read(io, Int16, imageDims)
    end
    img = AxisArray(permutedims(data, [2,1,3]),
                    Axis{:A}(Ranges.linspace(1u"mm", imageDims[2]*spacings[2]u"mm", imageDims[2])),
                    Axis{:R}(Ranges.linspace(1u"mm", imageDims[1]*spacings[1]u"mm", imageDims[1])),
                    Axis{:S}(Ranges.linspace(1u"mm", imageDims[3]*spacings[3]u"mm", imageDims[3])))
end
