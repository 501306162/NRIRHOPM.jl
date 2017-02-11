# locally register NIfTI format to FileIO
add_format(format"NIfTI", (), ".nii")
add_loader(format"NIfTI", :NIfTI)
add_saver(format"NIfTI", :NIfTI)

function FileIO.load(f::File{format"NIfTI"}; kwargs...)
    nii = niread(filename(f), kwargs...)
    data = squeeze(nii.raw, (4,5,6))
    spacings = nii.header.pixdim[2:4]
    imageDims = nii.header.dim[2:4]
    img = AxisArray(data,
                    Axis{:x}(Ranges.linspace(1u"mm", imageDims[1]*spacings[1]u"mm", imageDims[1])),
                    Axis{:y}(Ranges.linspace(1u"mm", imageDims[2]*spacings[2]u"mm", imageDims[2])),
                    Axis{:z}(Ranges.linspace(1u"mm", imageDims[3]*spacings[3]u"mm", imageDims[3])))
end

FileIO.save(f::File{format"NIfTI"}, data) = niwrite(filename(f), data)
