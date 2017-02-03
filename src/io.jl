function read(path)
    nii = niread(path)
    data = squeeze(nii.raw, (4,5,6))
    spacings = nii.header.pixdim[2:4]
    imageDims = nii.header.dim[2:4]
    img = AxisArray(data,
                    Axis{:x}(Ranges.linspace(1u"mm", imageDims[1]*spacings[1]u"mm", imageDims[1])),
                    Axis{:y}(Ranges.linspace(1u"mm", imageDims[2]*spacings[2]u"mm", imageDims[2])),
                    Axis{:z}(Ranges.linspace(1u"mm", imageDims[3]*spacings[3]u"mm", imageDims[3])))
end
