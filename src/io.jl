const mm = u"mm"

function read(path)
    nii = niread("path")
    data = squeeze(nii.raw, (4,5,6))
    spacings = nii.header.pixdim[2:4]
    imageDims = nii.header.dim[2:4]
    img = AxisArray(data,
                    Axis{:x}(1mm:spacings[1]mm:spacings[1]*imageDims[1]mm),
                    Axis{:y}(1mm:spacings[2]mm:spacings[2]*imageDims[2]mm),
                    Axis{:z}(1mm:spacings[3]mm:spacings[3]*imageDims[3]mm))
end
