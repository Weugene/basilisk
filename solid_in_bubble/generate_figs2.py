# state file generated using paraview version 5.9.0
# import the simple module from the paraview
from __future__ import annotations

from paraview.simple import *

# disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView("RenderView")
renderView1.ViewSize = [1380, 1156]
renderView1.InteractionMode = "2D"
renderView1.AxesGrid = "GridAxes3DActor"
renderView1.OrientationAxesVisibility = 0
renderView1.StereoType = "Crystal Eyes"
renderView1.CameraPosition = [0.0038110339995781, -0.013443640982182625, 10000.0]
renderView1.CameraFocalPoint = [0.0038110339995781, -0.013443640982182625, 0.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 0.40058869992898755
renderView1.BackEnd = "OSPRay raycaster"
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name="Layout #1")
layout1.AssignView(0, renderView1)
layout1.SetSize(1380, 1156)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
input = PVDReader(
    registrationName="input",
    FileName="/home/e.sharaborin/basilisk/work/solid_in_bubble/bpmuflinear_jmax=9_Re=100_Ca=0.01_Rratio=2.pvd",
)
input.CellArrays = ["f", "fs"]

# create a new 'Cell Data to Point Data'
cellDatatoPointData = CellDatatoPointData(registrationName="CellDatatoPointData", Input=input)
cellDatatoPointData.CellDataArraytoprocess = ["f", "fs"]

# create a new 'Resample To Image'
resampleToImage = ResampleToImage(registrationName="ResampleToImage", Input=cellDatatoPointData)
resampleToImage.UseInputBounds = 0
resampleToImage.SamplingDimensions = [512, 512, 1]
resampleToImage.SamplingBounds = [-0.48, 0.48, -0.5, 0.5, 0.0, 0.0]

# create a new 'Contour'
contour3 = Contour(registrationName="Contour3", Input=resampleToImage)
contour3.ContourBy = ["POINTS", "f"]
contour3.Isosurfaces = [0.5]
contour3.PointMergeMethod = "Uniform Binning"

# create a new 'Threshold'
threshold4 = Threshold(registrationName="Threshold4", Input=resampleToImage)
threshold4.Scalars = ["POINTS", "fs"]
threshold4.ThresholdRange = [0.45, 1.0]

# create a new 'Threshold'
threshold3 = Threshold(registrationName="Threshold3", Input=resampleToImage)
threshold3.Scalars = ["POINTS", "fs"]
threshold3.ThresholdRange = [0.0, 0.5]

# create a new 'Contour'
contour4 = Contour(registrationName="Contour4", Input=resampleToImage)
contour4.ContourBy = ["POINTS", "fs"]
contour4.Isosurfaces = [0.5]
contour4.PointMergeMethod = "Uniform Binning"

# create a new 'Clip'
clip = Clip(registrationName="Clip", Input=resampleToImage)
clip.ClipType = "Box"
clip.HyperTreeGridClipper = "Plane"
clip.Scalars = ["POINTS", "Solid"]
clip.Value = 1.5000000000000002

# init the 'Box' selected for 'ClipType'
clip.ClipType.Position = [0.0, -0.07, -0.05]
clip.ClipType.Length = [0.15, 0.14, 0.1]

# create a new 'Contour'
contour2 = Contour(registrationName="Contour2", Input=clip)
contour2.ContourBy = ["POINTS", "f"]
contour2.Isosurfaces = [0.5]
contour2.PointMergeMethod = "Uniform Binning"

# create a new 'Threshold'
threshold1 = Threshold(registrationName="Threshold1", Input=clip)
threshold1.Scalars = ["POINTS", "fs"]
threshold1.ThresholdRange = [0.0, 0.5]

# create a new 'Threshold'
threshold2 = Threshold(registrationName="Threshold2", Input=clip)
threshold2.Scalars = ["POINTS", "fs"]
threshold2.ThresholdRange = [0.45, 1.0]

# create a new 'Contour'
contour1 = Contour(registrationName="Contour1", Input=clip)
contour1.ContourBy = ["POINTS", "fs"]
contour1.Isosurfaces = [0.5]
contour1.PointMergeMethod = "Uniform Binning"

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from contour2
contour2Display = Show(contour2, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
contour2Display.Representation = "Surface"
contour2Display.AmbientColor = [0.0, 0.0, 1.0]
contour2Display.ColorArrayName = ["POINTS", ""]
contour2Display.DiffuseColor = [0.0, 0.0, 1.0]
contour2Display.LineWidth = 10.0
contour2Display.SelectTCoordArray = "None"
contour2Display.SelectNormalArray = "None"
contour2Display.SelectTangentArray = "None"
contour2Display.Position = [0.165, 0.25, 0.0]
contour2Display.Scale = [2.1, 2.1, 1.0]
contour2Display.OSPRayScaleArray = "f"
contour2Display.OSPRayScaleFunction = "PiecewiseFunction"
contour2Display.SelectOrientationVectors = "None"
contour2Display.ScaleFactor = 0.025049885362386705
contour2Display.SelectScaleArray = "f"
contour2Display.GlyphType = "Arrow"
contour2Display.GlyphTableIndexArray = "f"
contour2Display.GaussianRadius = 0.0012524942681193352
contour2Display.SetScaleArray = ["POINTS", "f"]
contour2Display.ScaleTransferFunction = "PiecewiseFunction"
contour2Display.OpacityArray = ["POINTS", "f"]
contour2Display.OpacityTransferFunction = "PiecewiseFunction"
contour2Display.DataAxesGrid = "GridAxesRepresentation"
contour2Display.PolarAxes = "PolarAxesRepresentation"

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour2Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour2Display.OpacityTransferFunction.Points = [
    0.5,
    0.0,
    0.5,
    0.0,
    0.5001220703125,
    1.0,
    0.5,
    0.0,
]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
contour2Display.PolarAxes.Translation = [0.165, 0.25, 0.0]
contour2Display.PolarAxes.Scale = [2.1, 2.1, 1.0]

# show data from contour1
contour1Display = Show(contour1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
contour1Display.Representation = "Surface"
contour1Display.AmbientColor = [0.0, 0.0, 0.0]
contour1Display.ColorArrayName = ["POINTS", ""]
contour1Display.DiffuseColor = [0.0, 0.0, 0.0]
contour1Display.LineWidth = 10.0
contour1Display.SelectTCoordArray = "None"
contour1Display.SelectNormalArray = "None"
contour1Display.SelectTangentArray = "None"
contour1Display.Position = [0.165, 0.25, 0.0]
contour1Display.Scale = [2.1, 2.1, 1.0]
contour1Display.OSPRayScaleArray = "fs"
contour1Display.OSPRayScaleFunction = "PiecewiseFunction"
contour1Display.SelectOrientationVectors = "None"
contour1Display.ScaleFactor = 0.012521313130855562
contour1Display.SelectScaleArray = "fs"
contour1Display.GlyphType = "Arrow"
contour1Display.GlyphTableIndexArray = "fs"
contour1Display.GaussianRadius = 0.000626065656542778
contour1Display.SetScaleArray = ["POINTS", "fs"]
contour1Display.ScaleTransferFunction = "PiecewiseFunction"
contour1Display.OpacityArray = ["POINTS", "fs"]
contour1Display.OpacityTransferFunction = "PiecewiseFunction"
contour1Display.DataAxesGrid = "GridAxesRepresentation"
contour1Display.PolarAxes = "PolarAxesRepresentation"

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [
    0.5,
    0.0,
    0.5,
    0.0,
    0.5001220703125,
    1.0,
    0.5,
    0.0,
]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
contour1Display.PolarAxes.Translation = [0.165, 0.25, 0.0]
contour1Display.PolarAxes.Scale = [2.1, 2.1, 1.0]

# show data from threshold1
threshold1Display = Show(threshold1, renderView1, "UnstructuredGridRepresentation")

# get color transfer function/color map for 'f'
fLUT = GetColorTransferFunction("f")
fLUT.RGBPoints = [
    0.0,
    1.0,
    1.0,
    1.0,
    0.5,
    1.0,
    1.0,
    1.0,
    1.0,
    0.4549019607843137,
    0.807843137254902,
    1.0,
]
fLUT.ColorSpace = "Step"
fLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'f'
fPWF = GetOpacityTransferFunction("f")
fPWF.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    0.3800622890658208,
    0.38717949390411377,
    0.5,
    0.0,
    1.0,
    1.0,
    0.5,
    0.0,
]
fPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
threshold1Display.Representation = "Surface"
threshold1Display.ColorArrayName = ["POINTS", "f"]
threshold1Display.LookupTable = fLUT
threshold1Display.Ambient = 1.0
threshold1Display.Diffuse = 0.0
threshold1Display.SelectTCoordArray = "None"
threshold1Display.SelectNormalArray = "None"
threshold1Display.SelectTangentArray = "None"
threshold1Display.Position = [0.165, 0.25, 0.0]
threshold1Display.Scale = [2.1, 2.1, 1.0]
threshold1Display.OSPRayScaleArray = "Solid"
threshold1Display.OSPRayScaleFunction = "PiecewiseFunction"
threshold1Display.SelectOrientationVectors = "None"
threshold1Display.ScaleFactor = 0.015000000596046299
threshold1Display.SelectScaleArray = "Solid"
threshold1Display.GlyphType = "Arrow"
threshold1Display.GlyphTableIndexArray = "Solid"
threshold1Display.GaussianRadius = 0.0007500000298023149
threshold1Display.SetScaleArray = ["POINTS", "Solid"]
threshold1Display.ScaleTransferFunction = "PiecewiseFunction"
threshold1Display.OpacityArray = ["POINTS", "Solid"]
threshold1Display.OpacityTransferFunction = "PiecewiseFunction"
threshold1Display.DataAxesGrid = "GridAxesRepresentation"
threshold1Display.PolarAxes = "PolarAxesRepresentation"
threshold1Display.ScalarOpacityFunction = fPWF
threshold1Display.ScalarOpacityUnitDistance = 0.01524508409063841
threshold1Display.OpacityArrayName = ["POINTS", "Solid"]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
threshold1Display.ScaleTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    2.845091279521759,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
threshold1Display.OpacityTransferFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    2.845091279521759,
    1.0,
    0.5,
    0.0,
]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
threshold1Display.PolarAxes.Translation = [0.165, 0.25, 0.0]
threshold1Display.PolarAxes.Scale = [2.1, 2.1, 1.0]

# show data from threshold2
threshold2Display = Show(threshold2, renderView1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
threshold2Display.Representation = "Surface"
threshold2Display.AmbientColor = [0.6196078431372549, 0.6196078431372549, 0.6196078431372549]
threshold2Display.ColorArrayName = ["POINTS", ""]
threshold2Display.DiffuseColor = [0.6196078431372549, 0.6196078431372549, 0.6196078431372549]
threshold2Display.Ambient = 1.0
threshold2Display.Diffuse = 0.0
threshold2Display.SelectTCoordArray = "None"
threshold2Display.SelectNormalArray = "None"
threshold2Display.SelectTangentArray = "None"
threshold2Display.Position = [0.165, 0.25, 0.0]
threshold2Display.Scale = [2.1, 2.1, 1.0]
threshold2Display.OSPRayScaleArray = "Solid"
threshold2Display.OSPRayScaleFunction = "PiecewiseFunction"
threshold2Display.SelectOrientationVectors = "None"
threshold2Display.ScaleFactor = 0.012328767031431199
threshold2Display.SelectScaleArray = "Solid"
threshold2Display.GlyphType = "Arrow"
threshold2Display.GlyphTableIndexArray = "Solid"
threshold2Display.GaussianRadius = 0.0006164383515715599
threshold2Display.SetScaleArray = ["POINTS", "Solid"]
threshold2Display.ScaleTransferFunction = "PiecewiseFunction"
threshold2Display.OpacityArray = ["POINTS", "Solid"]
threshold2Display.OpacityTransferFunction = "PiecewiseFunction"
threshold2Display.DataAxesGrid = "GridAxesRepresentation"
threshold2Display.PolarAxes = "PolarAxesRepresentation"
threshold2Display.ScalarOpacityUnitDistance = 0.01173782462695837
threshold2Display.OpacityArrayName = ["POINTS", "Solid"]

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
threshold2Display.ScaleTransferFunction.Points = [
    1.5183730914020697,
    0.0,
    0.5,
    0.0,
    2.7808219178082254,
    1.0,
    0.5,
    0.0,
]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
threshold2Display.OpacityTransferFunction.Points = [
    1.5183730914020697,
    0.0,
    0.5,
    0.0,
    2.7808219178082254,
    1.0,
    0.5,
    0.0,
]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
threshold2Display.PolarAxes.Translation = [0.165, 0.25, 0.0]
threshold2Display.PolarAxes.Scale = [2.1, 2.1, 1.0]

# show data from threshold3
threshold3Display = Show(threshold3, renderView1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
threshold3Display.Representation = "Surface"
threshold3Display.ColorArrayName = ["POINTS", "f"]
threshold3Display.LookupTable = fLUT
threshold3Display.Ambient = 1.0
threshold3Display.Diffuse = 0.0
threshold3Display.SelectTCoordArray = "None"
threshold3Display.SelectNormalArray = "None"
threshold3Display.SelectTangentArray = "None"
threshold3Display.OSPRayScaleArray = "f"
threshold3Display.OSPRayScaleFunction = "PiecewiseFunction"
threshold3Display.SelectOrientationVectors = "None"
threshold3Display.ScaleFactor = 0.1
threshold3Display.SelectScaleArray = "None"
threshold3Display.GlyphType = "Arrow"
threshold3Display.GlyphTableIndexArray = "None"
threshold3Display.GaussianRadius = 0.005
threshold3Display.SetScaleArray = ["POINTS", "f"]
threshold3Display.ScaleTransferFunction = "PiecewiseFunction"
threshold3Display.OpacityArray = ["POINTS", "f"]
threshold3Display.OpacityTransferFunction = "PiecewiseFunction"
threshold3Display.DataAxesGrid = "GridAxesRepresentation"
threshold3Display.PolarAxes = "PolarAxesRepresentation"
threshold3Display.ScalarOpacityFunction = fPWF
threshold3Display.ScalarOpacityUnitDistance = 0.021784972811574
threshold3Display.OpacityArrayName = ["POINTS", "f"]

# show data from threshold4
threshold4Display = Show(threshold4, renderView1, "UnstructuredGridRepresentation")

# trace defaults for the display properties.
threshold4Display.Representation = "Surface"
threshold4Display.AmbientColor = [0.6196078431372549, 0.6196078431372549, 0.6196078431372549]
threshold4Display.ColorArrayName = ["POINTS", ""]
threshold4Display.DiffuseColor = [0.6196078431372549, 0.6196078431372549, 0.6196078431372549]
threshold4Display.Ambient = 1.0
threshold4Display.Diffuse = 0.0
threshold4Display.SelectTCoordArray = "None"
threshold4Display.SelectNormalArray = "None"
threshold4Display.SelectTangentArray = "None"
threshold4Display.OSPRayScaleArray = "f"
threshold4Display.OSPRayScaleFunction = "PiecewiseFunction"
threshold4Display.SelectOrientationVectors = "None"
threshold4Display.ScaleFactor = 0.012328767031431199
threshold4Display.SelectScaleArray = "None"
threshold4Display.GlyphType = "Arrow"
threshold4Display.GlyphTableIndexArray = "None"
threshold4Display.GaussianRadius = 0.0006164383515715599
threshold4Display.SetScaleArray = ["POINTS", "f"]
threshold4Display.ScaleTransferFunction = "PiecewiseFunction"
threshold4Display.OpacityArray = ["POINTS", "f"]
threshold4Display.OpacityTransferFunction = "PiecewiseFunction"
threshold4Display.DataAxesGrid = "GridAxesRepresentation"
threshold4Display.PolarAxes = "PolarAxesRepresentation"
threshold4Display.ScalarOpacityUnitDistance = 0.01173782462695837
threshold4Display.OpacityArrayName = ["POINTS", "f"]

# show data from contour3
contour3Display = Show(contour3, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
contour3Display.Representation = "Surface"
contour3Display.AmbientColor = [0.0, 0.0, 1.0]
contour3Display.ColorArrayName = ["POINTS", ""]
contour3Display.DiffuseColor = [0.0, 0.0, 1.0]
contour3Display.LineWidth = 5.0
contour3Display.SelectTCoordArray = "None"
contour3Display.SelectNormalArray = "None"
contour3Display.SelectTangentArray = "None"
contour3Display.OSPRayScaleArray = "f"
contour3Display.OSPRayScaleFunction = "PiecewiseFunction"
contour3Display.SelectOrientationVectors = "None"
contour3Display.ScaleFactor = 0.025049885362386705
contour3Display.SelectScaleArray = "f"
contour3Display.GlyphType = "Arrow"
contour3Display.GlyphTableIndexArray = "f"
contour3Display.GaussianRadius = 0.0012524942681193352
contour3Display.SetScaleArray = ["POINTS", "f"]
contour3Display.ScaleTransferFunction = "PiecewiseFunction"
contour3Display.OpacityArray = ["POINTS", "f"]
contour3Display.OpacityTransferFunction = "PiecewiseFunction"
contour3Display.DataAxesGrid = "GridAxesRepresentation"
contour3Display.PolarAxes = "PolarAxesRepresentation"

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour3Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour3Display.OpacityTransferFunction.Points = [
    0.5,
    0.0,
    0.5,
    0.0,
    0.5001220703125,
    1.0,
    0.5,
    0.0,
]

# show data from contour4
contour4Display = Show(contour4, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
contour4Display.Representation = "Surface"
contour4Display.AmbientColor = [0.0, 0.0, 0.0]
contour4Display.ColorArrayName = ["POINTS", ""]
contour4Display.DiffuseColor = [0.0, 0.0, 0.0]
contour4Display.LineWidth = 5.0
contour4Display.SelectTCoordArray = "None"
contour4Display.SelectNormalArray = "None"
contour4Display.SelectTangentArray = "None"
contour4Display.OSPRayScaleArray = "fs"
contour4Display.OSPRayScaleFunction = "PiecewiseFunction"
contour4Display.SelectOrientationVectors = "None"
contour4Display.ScaleFactor = 0.012521313130855562
contour4Display.SelectScaleArray = "fs"
contour4Display.GlyphType = "Arrow"
contour4Display.GlyphTableIndexArray = "fs"
contour4Display.GaussianRadius = 0.000626065656542778
contour4Display.SetScaleArray = ["POINTS", "fs"]
contour4Display.ScaleTransferFunction = "PiecewiseFunction"
contour4Display.OpacityArray = ["POINTS", "fs"]
contour4Display.OpacityTransferFunction = "PiecewiseFunction"
contour4Display.DataAxesGrid = "GridAxesRepresentation"
contour4Display.PolarAxes = "PolarAxesRepresentation"

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour4Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour4Display.OpacityTransferFunction.Points = [
    0.5,
    0.0,
    0.5,
    0.0,
    0.5001220703125,
    1.0,
    0.5,
    0.0,
]

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(input)
# ----------------------------------------------------------------


if __name__ == "__main__":
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory="extracts")
