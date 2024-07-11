import logging
import os
from typing import Annotated, Optional
import qt
import vtk
from vtk.util.numpy_support import numpy_to_vtk #type: ignore
import random
import time
import uuid
import numpy as np 


# import Simulations.SOFASimulationMulti as multi

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)


from SofaEnvironment import Sofa
from SlicerSofa import SlicerSofaLogic

#from slicer import vtkMRMLIGTLConnectorNode
from slicer import vtkMRMLMarkupsFiducialNode
from slicer import vtkMRMLMarkupsLineNode
from slicer import vtkMRMLMarkupsNode
from slicer import vtkMRMLMarkupsROINode
from slicer import vtkMRMLModelNode
from slicer import vtkMRMLSequenceBrowserNode
from slicer import vtkMRMLSequenceNode
from slicer import vtkMRMLGridTransformNode
#from slicer import vtkMRMLPlotDataNode




#
# AirwaySimulation
#


class AirwaySimulation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Airway Simulation")
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []
        self.parent.contributors = ["Quinn Williams(Brigham and Women's Hospital)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""This is a module developed to simulate lung movement for aiding broncoscopies.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This module was developed by Quinn Williams with help from Franklin King, Rafael Palomar, and Steve Pieper.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # AirwaySimulation1
    '''SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="AirwaySimulation",
        sampleName="Lung Mesh",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "AirwaySimulation1.png"),
        # Download URL and target file name
        uris="https://github.com/Quilliams85/Lung-Simulation-SNR-Lab/mesh/",
        fileNames="lunglow.vtk",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        nodeNames='RightLung',
        loadFileType='ModelFile',
        nodeNames="Lung Mesh",
    )'''

    # AirwaySimulation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="AirwaySimulation",
        sampleName="AirwaySimulation2",
        thumbnailFileName=os.path.join(iconsPath, "AirwaySimulation2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="AirwaySimulation2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="AirwaySimulation2",
    )

#
# AirwaySimulationParameterNode
#


@parameterNodeWrapper
class AirwaySimulationParameterNode:
    """
    The parameters needed by module.
    """
    #Simulation data
    modelNode: vtkMRMLModelNode
    ribsModelNode: vtkMRMLModelNode
    transformationNode: vtkMRMLGridTransformNode
    boundaryROI: vtkMRMLMarkupsROINode
    gravityVector: vtkMRMLMarkupsLineNode
    gravityMagnitude: int
    sequenceNode: vtkMRMLSequenceNode
    sequenceBrowserNode: vtkMRMLSequenceBrowserNode
    #Simulation control
    dt: float
    totalSteps: int
    currentStep: int
    simulationRunning: bool
    breathingPeriod: float
    breathingForce: float
    conversionFactor: int
    poissonRatio: float
    youngsModulus: float

    def getBoundaryROI(self):

        if self.boundaryROI is None:
            return [0.0]*6

        center = [0]*3
        self.boundaryROI.GetCenter(center)
        size = self.boundaryROI.GetSize()

        # Calculate min and max RAS bounds from center and size
        R_min = center[0] - size[0] / 2
        R_max = center[0] + size[0] / 2
        A_min = center[1] - size[1] / 2
        A_max = center[1] + size[1] / 2
        S_min = center[2] - size[2] / 2
        S_max = center[2] + size[2] / 2

        # Return the two opposing bounds corners
        # First corner: (minL, minP, minS), Second corner: (maxL, maxP, maxS)
        return [R_min, A_min, S_min, R_max, A_max, S_max]
    
    #grid transform update function
    


    def getGravityVector(self):

        if self.gravityVector is None:
            return [0.0]*3

        p1 = self.gravityVector.GetNthControlPointPosition(0)
        p2 = self.gravityVector.GetNthControlPointPosition(1)
        gravity_vector = np.array([p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]])
        magnitude = np.linalg.norm(gravity_vector)
        normalized_gravity_vector = gravity_vector / magnitude if magnitude != 0 else gravity_vector

        return normalized_gravity_vector*self.gravityMagnitude
    

    def getSurfacePointsArray(self, polydata):
        points = polydata.GetPolyData().GetPoints()
        num_points = points.GetNumberOfPoints()
        vertices = []
        for i in range(num_points):
            p = points.GetPoint(i)
            for a in p:
                a *= 1.0
            vertices.append(p)
        return vertices


    def getModelPointsArray(self, targetNode):
        """
        Convert the point positions from the VTK model to a Python list.
        """
        # Get the unstructured grid from the model node
        unstructured_grid = targetNode.GetUnstructuredGrid()

        if unstructured_grid == None:
            print('Wrong Mesh type, no unstructured grid!')

        # Extract point data from the unstructured grid
        points = unstructured_grid.GetPoints()
        num_points = points.GetNumberOfPoints()

        # Convert the VTK points to a list
        point_coords = []
        for i in range(num_points):
            #for j in points.GetPoint(i):
                #j /= self.conversionFactor
            point_coords.append(points.GetPoint(i))
        return point_coords

    def getModelCellsArray(self):
        """
        Convert the cell connectivity from the VTK model to a Python list.
        """
        # Get the unstructured grid from the model node
        unstructured_grid = self.modelNode.GetUnstructuredGrid()

        # Extract cell data from the unstructured grid
        cells = unstructured_grid.GetCells()
        cell_array = vtk.util.numpy_support.vtk_to_numpy(cells.GetData())

        # The first integer in each cell entry is the number of points per cell (always 4 for tetrahedra)
        # Followed by the point indices
        num_cells = unstructured_grid.GetNumberOfCells()
        cell_connectivity = []

        # Fill the cell connectivity list
        idx = 0
        for i in range(num_cells):
            num_points = cell_array[idx]  # Should always be 4 for tetrahedra
            cell_connectivity.append(cell_array[idx+1:idx+1+num_points].tolist())
            idx += num_points + 1

        return cell_connectivity


#
# AirwaySimulationWidget
#


class AirwaySimulationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self.parameterNode = None
        self.parameterNodeGuiTag = None
        self.timer = qt.QTimer(parent)
        self.timer.timeout.connect(self.simulationStep)

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AirwaySimulation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = AirwaySimulationLogic()

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        #self.logic.getParameterNode().AddObserver(vtk.vtkCommand.ModifiedEvent, self.printsomething())
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.startSimulationPushButton.connect("clicked()", self.startSimulation)
        self.ui.stopSimulationPushButton.connect("clicked()", self.stopSimulation)
        self.ui.resetSimulationPushButton.connect("clicked()", self.resetSimulation)
        self.ui.addBoundaryROIPushButton.connect("clicked()", self.logic.addBoundaryROI)
        self.ui.addGravityVectorPushButton.connect("clicked()", self.logic.addGravityVector)
        self.ui.addRecordingSequencePushButton.connect("clicked()", self.logic.addRecordingSequence)


        self.logic.getParameterNode().conversionFactor = 1

        self.initializeParameterNode()
        self.logic.getParameterNode().AddObserver(vtk.vtkCommand.ModifiedEvent, self.updateSimulationGUI)



    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.timer.stop()
        self.logic.stopSimulation()
        self.logic.clean()
        self.removeObservers()

    def enter(self) -> None:
        # """Called each time the user opens this module."""
        # # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
            self.parameterNodeGuiTag = None

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        if self.logic:
            self.setParameterNode(self.logic.getParameterNode())
            self.logic.resetParameterNode()

    def setParameterNode(self, inputParameterNode: Optional[AirwaySimulationParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if self.parameterNode:
            self.parameterNode.disconnectGui(self.parameterNodeGuiTag)
        self.parameterNode = inputParameterNode
        if self.parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self.parameterNodeGuiTag = self.parameterNode.connectGui(self.ui)

    def updateSimulationGUI(self, caller, event):
        """This enables/disables the simulation buttons according to the state of the parameter node"""
        self.ui.startSimulationPushButton.setEnabled(not self.logic.isSimulationRunning and
                                                     self.logic.getParameterNode().modelNode is not None)
        self.ui.stopSimulationPushButton.setEnabled(self.logic.isSimulationRunning)

    def startSimulation(self):
        if self.parameterNode.ribsModelNode != None:
            print('Using Ribs Mesh')
        else:
            print('No Ribs Mesh Selected')
        self.logic.dt = self.ui.dtSpinBox.value
        self.logic._totalSteps = self.ui.totalStepsSpinBox.value
        self.logic.currentStep = self.ui.currentStepSpinBox.value
        self.parameterNode.breathingPeriod = self.ui.breathingPeriodSlider.value
        self.parameterNode.breathingForce = self.ui.breathingForceSlider.value
        self.parameterNode.poissonRatio = self.ui.poissonRatioSpinBox.value
        self.parameterNode.youngsModulus = self.ui.youngsModulusSpinBox.value
        self.logic.startSimulation()
        self.timer.start(0) #This timer drives the simulation updates

    def stopSimulation(self):
        self.timer.stop()
        self.logic.stopSimulation()

    def resetSimulation(self):
            self.timer.stop()
            self.ui.currentStepSpinBox.value = 0
            self.logic.resetSimulation()

    def simulationStep(self):
       self.logic.simulationStep(self.parameterNode)
       self.ui.currentStepSpinBox.value += 1
       self.ui.systemForceBar.value = self.logic.systemForce


#
# AirwaySimulationLogic
#


class AirwaySimulationLogic(SlicerSofaLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        super().__init__()
        self.connectionStatus = 0
        self.boxROI = None
        self.mouseInteractor = None
        self.startup = True
        self.systemForce = 0
        self.probeDimension = 40

    def updateSofa(self, parameterNode) -> None:
        if parameterNode is not None:
            self.BoxROI.box = [parameterNode.getBoundaryROI()]
        if parameterNode.gravityVector is not None:
            self.rootNode.gravity = parameterNode.getGravityVector()

    def updateMRML(self, parameterNode) -> None:
        #update lung mesh
        scaled_positions = self.mechanicalObject.position.array() * self.getParameterNode().conversionFactor
        points_vtk = numpy_to_vtk(num_array=scaled_positions, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(points_vtk)
        parameterNode.modelNode.GetUnstructuredGrid().SetPoints(vtk_points)

        #pass back ribs mesh
        points_vtk1 = numpy_to_vtk(num_array = self.ribsMO.position.array(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points1 = vtk.vtkPoints()
        vtk_points1.SetData(points_vtk1)
        parameterNode.ribsModelNode.GetPolyData().SetPoints(vtk_points1)


        self.systemForce = np.linalg.norm(self.femMechanicalObject.force.value)


        #Update Grid Transform
        displacementArray = slicer.util.arrayFromModelPointData(parameterNode.modelNode, "Displacement")
        displacementArray[:] = (self.mechanicalObject.position - self.mechanicalObject.rest_position) 
        #displacementArray *= np.array([-1, -1, 1])
        slicer.util.arrayFromModelPointsModified(parameterNode.modelNode)
        self.probeFilter.Update()
        probeImage = self.probeFilter.GetOutputDataObject(0)
        probeVTKArray = probeImage.GetPointData().GetArray("Displacement")
        probeArray = vtk.util.numpy_support.vtk_to_numpy(probeVTKArray)
        probeArrayShape = (self.probeDimension,self.probeDimension,self.probeDimension,3)
        probeArray = probeArray.reshape(probeArrayShape)
        gridArray = slicer.util.arrayFromGridTransform(self.displacementGridNode)
        gridArray[:] = -1. * probeArray
        slicer.util.arrayFromGridTransformModified(self.displacementGridNode)


        #update von mises
        stressArray = slicer.util.arrayFromModelCellData(parameterNode.modelNode, "VonMisesStress")
        stressArray[:] = self.forceField.vonMisesPerElement.array()
        slicer.util.arrayFromModelCellDataModified(parameterNode.modelNode, "VonMisesStress")

    def getParameterNode(self):
        return AirwaySimulationParameterNode(super().getParameterNode())

    def resetParameterNode(self):
        if self.getParameterNode():
            self.getParameterNode().modelNode = None
            self.getParameterNode().ribsModelNode = None
            self.getParameterNode().boundaryROI = None
            self.getParameterNode().gravityVector = None
            self.getParameterNode().sequenceNode = None
            self.getParameterNode().sequenceBrowserNode = None
            self.getParameterNode().dt = 0.01
            self.getParameterNode().currentStep = 0
            self.getParameterNode().totalSteps = -1

    def getSimulationController(self):
        return self.simulationController
    
    def getTransformation(self, currentgrid):
        init_points = np.array(self.initialgrid)
        return currentgrid - init_points
    
    def numpy_to_vtkImageData(self, np_array):
        # Ensure the numpy array is contiguous
        np_array = np.ascontiguousarray(np_array)

        # Create a vtkImageImport object
        importer = vtk.vtkImageImport()

        # Set the data attributes
        importer.CopyImportVoidPointer(np_array, np_array.nbytes)
        importer.SetDataScalarTypeToUnsignedChar()  # Use appropriate data type
        importer.SetNumberOfScalarComponents(np_array.shape[2] if np_array.ndim == 3 else 1)

        # Set the dimensions of the data
        importer.SetDataExtent(0, np_array.shape[1] - 1, 0, np_array.shape[0] - 1, 0, 0)
        importer.SetWholeExtent(0, np_array.shape[1] - 1, 0, np_array.shape[0] - 1, 0, 0)

        # If the array has three dimensions (like for RGB images), adjust the Z extent
        if np_array.ndim == 3:
            importer.SetDataExtent(0, np_array.shape[1] - 1, 0, np_array.shape[0] - 1, 0, np_array.shape[2] - 1)
            importer.SetWholeExtent(0, np_array.shape[1] - 1, 0, np_array.shape[0] - 1, 0, np_array.shape[2] - 1)

        # Update the importer to generate the vtkImageData
        importer.Update()

        # Get the vtkImageData object
        vtk_image_data = importer.GetOutput()

        return vtk_image_data


    def startSimulation(self) -> None:
        sequenceNode = self.getParameterNode().sequenceNode
        browserNode = self.getParameterNode().sequenceBrowserNode
        modelNode = self.getParameterNode().modelNode
        ribsModeNode = self.getParameterNode().ribsModelNode

        # Synchronize and set up the sequence browser node
        if None not in [sequenceNode, browserNode, modelNode]:
            browserNode.AddSynchronizedSequenceNodeID(sequenceNode.GetID())
            browserNode.AddProxyNode(modelNode, sequenceNode, False)
            browserNode.SetRecording(sequenceNode, True)
            browserNode.SetRecordingActive(True)
            

        #if it's the first time running simulation, save the initial mesh points for reset
        if self.startup:
            self.startup = False
            self.initialgrid = self.getParameterNode().getModelPointsArray(self.getParameterNode().modelNode)

        displacementVTKArray = vtk.vtkFloatArray()
        displacementVTKArray.SetNumberOfComponents(3)
        displacementVTKArray.SetNumberOfTuples(modelNode.GetUnstructuredGrid().GetNumberOfPoints())
        displacementVTKArray.SetName("Displacement")
        modelNode.GetUnstructuredGrid().GetPointData().AddArray(displacementVTKArray)

        probeGrid = vtk.vtkImageData()
        probeGrid.SetDimensions(self.probeDimension, self.probeDimension, self.probeDimension)
        probeGrid.AllocateScalars(vtk.VTK_DOUBLE, 1)
        meshBounds = [0]*6
        modelNode.GetRASBounds(meshBounds)
        probeGrid.SetOrigin(meshBounds[0], meshBounds[2], meshBounds[4])
        probeSize = (meshBounds[1] - meshBounds[0], meshBounds[3] - meshBounds[2], meshBounds[5] - meshBounds[4]) * 2
        probeGrid.SetSpacing(probeSize[0]/self.probeDimension, probeSize[1]/self.probeDimension, probeSize[2]/self.probeDimension)

        self.probeFilter = vtk.vtkProbeFilter()
        self.probeFilter.SetInputData(probeGrid)
        self.probeFilter.SetSourceData(modelNode.GetUnstructuredGrid())
        self.probeFilter.SetPassPointArrays(True)
        self.probeFilter.Update()

        probeImage = self.probeFilter.GetOutputDataObject(0)
        probeArray = vtk.util.numpy_support.vtk_to_numpy(probeImage.GetPointData().GetArray("Displacement"))
        probeArray = np.reshape(probeArray, (self.probeDimension,self.probeDimension,self.probeDimension,3))
        self.displacementGridNode = self.addGridTransformFromArray(probeArray, name="Displacement")
        self.displacementGrid = self.displacementGridNode.GetTransformFromParent().GetDisplacementGrid()
        # TODO: next two lines should be in ijkToRAS of grid node
        self.displacementGrid.SetOrigin(probeImage.GetOrigin())
        self.displacementGrid.SetSpacing(probeImage.GetSpacing())

        #setup von mises
        stressVTKArray = vtk.vtkFloatArray()
        stressVTKArray.SetNumberOfValues(modelNode.GetUnstructuredGrid().GetNumberOfCells())
        stressVTKArray.SetName("VonMisesStress")
        modelNode.GetUnstructuredGrid().GetCellData().AddArray(stressVTKArray)

        


        super().startSimulation(self.getParameterNode())
        self._simulationRunning = True
        self.getParameterNode().Modified()

    def stopSimulation(self) -> None:
        super().stopSimulation()
        self._simulationRunning = False
        browserNode = self.getParameterNode().sequenceBrowserNode
        if browserNode is not None:
            browserNode.SetRecordingActive(False)
        self.getParameterNode().Modified()

    def resetSimulation(self) -> None:
        self.currentStep = 0
        self.totalSteps = 0
        #reset mesh
        points_vtk = numpy_to_vtk(num_array=self.initialgrid, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(points_vtk)
        self.getParameterNode().modelNode.GetUnstructuredGrid().SetPoints(vtk_points)
        #reset simulation
        self.stopSimulation()
        self.clean()
        self.createScene(self.getParameterNode())
    


    def onModelNodeModified(self, caller, event) -> None:
        if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
            self.getParameterNode().modelNode.GetUnstructuredGrid().SetPoints(caller.GetPolyData().GetPoints())
        elif self.getParameterNode().modelNode.GetPolyData() is not None:
            self.getParameterNode().modelNode.GetPolyData().SetPoints(caller.GetPolyData().GetPoints())

    def addBoundaryROI(self) -> None:
        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode")
        mesh = None
        bounds = None

        if self.getParameterNode().modelNode is not None:
            if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().modelNode.GetUnstructuredGrid()
            elif self.getParameterNode().modelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().modelNode.GetPolyData()

        if mesh is not None:
            bounds = mesh.GetBounds()
            center = [(bounds[0] + bounds[1])/2.0, (bounds[2] + bounds[3])/2.0, (bounds[4] + bounds[5])/2.0]
            size = [abs(bounds[1] - bounds[0])/2.0, abs(bounds[3] - bounds[2])/2.0, abs(bounds[5] - bounds[4])/2.0]
            roiNode.SetXYZ(center)
            roiNode.SetRadiusXYZ(size[0], size[1], size[2])

        self.getParameterNode().boundaryROI = roiNode

    def getTransformBounds(self):
        mesh = None
        bounds = None

        if self.getParameterNode().modelNode is not None:
            if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().modelNode.GetUnstructuredGrid()
            elif self.getParameterNode().modelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().modelNode.GetPolyData()

        if mesh is not None:
            bounds = mesh.GetBounds()
            center = [(bounds[0] + bounds[1])/2.0, (bounds[2] + bounds[3])/2.0, (bounds[4] + bounds[5])/2.0]
            size = [abs(bounds[1] - bounds[0])/2.0, abs(bounds[3] - bounds[2])/2.0, abs(bounds[5] - bounds[4])/2.0]

        return None

    def addGravityVector(self) -> None:
        # Create a new line node for the gravity vector
        gravityVector = slicer.vtkMRMLMarkupsLineNode()
        gravityVector.SetName("Gravity")
        mesh = None

        # Check if there is a model node set in the parameter node and get its mesh
        if self.getParameterNode().modelNode is not None:
            if self.getParameterNode().modelNode.GetUnstructuredGrid() is not None:
                mesh = self.getParameterNode().modelNode.GetUnstructuredGrid()
            elif self.getParameterNode().modelNode.GetPolyData() is not None:
                mesh = self.getParameterNode().modelNode.GetPolyData()

        # If a mesh is found, compute its bounding box and center
        if mesh is not None:
            bounds = mesh.GetBounds()

            # Calculate the center of the bounding box
            center = [(bounds[0] + bounds[1])/2.0, (bounds[2] + bounds[3])/2.0, (bounds[4] + bounds[5])/2.0]

            # Calculate the vector's start and end points along the Y-axis, centered on the bounding box
            startPoint = [center[0], bounds[2], center[2]]  # Start at the bottom of the bounding box
            endPoint = [center[0], bounds[3], center[2]]  # End at the top of the bounding box

            # Adjust the start and end points to center the vector in the bounding box
            vectorLength = endPoint[1] - startPoint[1]
            midPoint = startPoint[1] + vectorLength / 2.0
            startPoint[1] = midPoint - vectorLength / 2.0
            endPoint[1] = midPoint + vectorLength / 2.0

            # Add control points to define the line
            gravityVector.AddControlPoint(vtk.vtkVector3d(startPoint))
            gravityVector.AddControlPoint(vtk.vtkVector3d(endPoint))

        # Add the gravity vector line node to the scene
        gravityVector = slicer.mrmlScene.AddNode(gravityVector)
        if gravityVector is not None:
            gravityVector.CreateDefaultDisplayNodes()

        self.getParameterNode().gravityVector = gravityVector



    def addRecordingSequence(self) -> None:

        browserNode = self.getParameterNode().sequenceBrowserNode
        modelNode = self.getParameterNode().modelNode

        # Ensure there is a sequence browser node; create if not present
        if browserNode is None:
            browserNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSequenceBrowserNode', "SOFA Simulation")
            browserNode.SetPlaybackActive(False)
            browserNode.SetRecordingActive(False)
            self.getParameterNode().sequenceBrowserNode = browserNode  # Update the parameter node reference

        sequenceNode = slicer.vtkMRMLSequenceNode()

        # Configure the sequence node based on the proxy model node
        if modelNode is not None:
            sequenceNodeName = modelNode.GetName() + "-Sequence"
            sequenceNode.SetName(sequenceNodeName)

        # Now add the configured sequence node to the scene
        slicer.mrmlScene.AddNode(sequenceNode)

        self.getParameterNode().sequenceNode = sequenceNode  # Update the parameter node reference

        # Configure index name and unit based on the master sequence node, if present
        masterSequenceNode = browserNode.GetMasterSequenceNode()
        if masterSequenceNode:
            sequenceNode.SetIndexName(masterSequenceNode.GetIndexName())
            sequenceNode.SetIndexUnit(masterSequenceNode.GetIndexUnit())

    def addGridTransformFromArray(self, narray, name="Grid Transform"):
        """Create a new grid transform node from content of a numpy array and add it to the scene.

        Voxels values are deep-copied, therefore if the numpy array
        is modified after calling this method, voxel values in the volume node will not change.
        :param narray: numpy array containing grid vectors.
        Must be [slices, rows, columns, 3]
        :param name: grid transform node name
        """
        if len(narray.shape) != 4 or narray.shape[3] != 3:
            raise RuntimeError("Need vector volume numpy array for grid transform")
        nodeClassName = "vtkMRMLGridTransformNode"
        gridNode = self.getParameterNode().transformationNode
        gridNode.CreateDefaultDisplayNodes()
        displacementGrid = gridNode.GetTransformFromParent().GetDisplacementGrid()
        arrayShape = narray.shape
        displacementGrid.SetDimensions(arrayShape[2], arrayShape[1], arrayShape[0])
        scalarType = vtk.util.numpy_support.get_vtk_array_type(narray.dtype)
        displacementGrid.AllocateScalars(scalarType, 3)
        displacementArray = slicer.util.arrayFromGridTransform(gridNode)
        displacementArray[:] = narray
        slicer.util.arrayFromGridTransformModified(gridNode)
        return gridNode

    
    def createScene(self, parameterNode) -> Sofa.Core.Node:
        from stlib3.scene import MainHeader, ContactHeader
        from stlib3.solver import DefaultSolver
        from stlib3.physics.deformable import ElasticMaterialObject
        from stlib3.physics.rigid import Floor
        from splib3.numerics import Vec3

        rootNode = Sofa.Core.Node()

        MainHeader(rootNode, plugins=[
            "Sofa.Component.IO.Mesh",
            "Sofa.Component.LinearSolver.Direct",
            "Sofa.Component.LinearSolver.Iterative",
            "Sofa.Component.Mapping.Linear",
            "Sofa.Component.Mass",
            "Sofa.Component.ODESolver.Backward",
            "Sofa.Component.Setting",
            "Sofa.Component.SolidMechanics.FEM.Elastic",
            "Sofa.Component.StateContainer",
            "Sofa.Component.Topology.Container.Dynamic",
            "Sofa.Component.Visual",
            "Sofa.GL.Component.Rendering3D",
            "Sofa.Component.AnimationLoop",
            "Sofa.Component.Collision.Detection.Algorithm",
            "Sofa.Component.Collision.Detection.Intersection",
            "Sofa.Component.Collision.Geometry",
            "Sofa.Component.Collision.Response.Contact",
            "Sofa.Component.Constraint.Lagrangian.Solver",
            "Sofa.Component.Constraint.Lagrangian.Correction",
            'Sofa.Component.Collision.Detection.Intersection',
            'Sofa.Component.Collision.Geometry',
            'Sofa.Component.Collision.Response.Contact',
            'Sofa.Component.Constraint.Projective',
            "Sofa.Component.LinearSystem",
            "Sofa.Component.MechanicalLoad",
            "MultiThreading",
            "Sofa.Component.SolidMechanics.Spring",
            "Sofa.Component.Constraint.Lagrangian.Model",
            "Sofa.Component.Mapping.NonLinear",
            "Sofa.Component.Topology.Container.Constant",
            "Sofa.Component.Topology.Mapping",
            "Sofa.Component.Topology.Container.Dynamic",
            "Sofa.Component.Engine.Select",
            "Sofa.Component.Constraint.Projective",
            "SofaIGTLink",
            "Sofa.Component.Mapping.NonLinear",
            'Sofa.Component.Topology.Container.Constant'

        ])

        rootNode.gravity = parameterNode.getGravityVector()
        rootNode.addObject('CollisionPipeline', name="CollisionPipeline")
        rootNode.addObject('BruteForceBroadPhase', name="BroadPhase")
        rootNode.addObject('BVHNarrowPhase', name="NarrowPhase")
        rootNode.addObject('DefaultContactManager', name="CollisionResponse", response="FrictionContactConstraint")
        rootNode.addObject('MinProximityIntersection', useLineLine=True, usePointPoint=True, alarmDistance=0.3 * 10, contactDistance=0.15 * 10, useLinePoint=True)


        rootNode.addObject('FreeMotionAnimationLoop', parallelODESolving=True, parallelCollisionDetectionAndFreeMotion=True)
        rootNode.addObject('GenericConstraintSolver', maxIterations=10, multithreading=True, tolerance=1.0e-3)

        femNode = rootNode.addChild('FEM')
        femNode.addObject('EulerImplicitSolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
        femNode.addObject('SparseLDLSolver', name="precond", template="CompressedRowSparseMatrixd", parallelInverseProduct=True)

        self.container = femNode.addObject('TetrahedronSetTopologyContainer', name="Container")
        self.container.position = parameterNode.getModelPointsArray(parameterNode.modelNode)
        self.container.tetrahedra = parameterNode.getModelCellsArray()



        femNode.addObject('TetrahedronSetTopologyModifier', name="Modifier")
        self.femMechanicalObject = femNode.addObject('MechanicalObject', name="mstate", template="Vec3d")
        self.mechanicalState = femNode.getMechanicalState()
        femNode.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=parameterNode.youngsModulus, poissonRatio=parameterNode.poissonRatio, method="large", computeVonMisesStress=2)
        femNode.addObject('MeshMatrixMass', totalMass=1)

        breathspeed = parameterNode.breathingPeriod / 100
        breathforce = parameterNode.breathingForce / 1000

        self.surfacePressure = femNode.addObject('SurfacePressureForceField', pressure=breathforce, pulseMode=True, pressureSpeed=breathspeed)
        self.forceField = femNode.getForceField(0)
        femNode.addObject('RestShapeSpringsForceField', stiffness=1, angularStiffness=1e-08)

        fixedROI = femNode.addChild('FixedROI')
        self.BoxROI = fixedROI.addObject('BoxROI', template="Vec3", box=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], drawBoxes=False,
                                          position="@../mstate.rest_position", name="FixedROI",
                                          computeTriangles=False, computeTetrahedra=False, computeEdges=False)
        fixedROI.addObject('FixedConstraint', indices="@FixedROI.indices")

        collisionNode = femNode.addChild('Collision')
        collisionNode.addObject('TriangleSetTopologyContainer', name="Container")
        collisionNode.addObject('TriangleSetTopologyModifier', name="Modifier")
        collisionNode.addObject('Tetra2TriangleTopologicalMapping', input="@../Container", output="@Container")
        collisionNode.addObject('TriangleCollisionModel', name="collisionModel", proximity=0.001, contactStiffness=20)
        collisionNode.addObject('PointCollisionModel')
        self.mechanicalObject = collisionNode.addObject('MechanicalObject', name='dofs', rest_position="@../mstate.rest_position")
        collisionNode.addObject('IdentityMapping', name='visualMapping')

        femNode.addObject('LinearSolverConstraintCorrection', linearSolver="@precond")

        from stlib3.physics.rigid import Cube, Sphere, Floor
        Floor(rootNode, name="plane", translation=[0.0, 0.0, 0.0], uniformScale=100)

        if parameterNode.ribsModelNode == None:
            return rootNode

        rib_points = parameterNode.getSurfacePointsArray(parameterNode.ribsModelNode)
        ribs = rootNode.addChild('ribs')
        
        self.ribsMO = ribs.addObject('MechanicalObject', name='dofs', position=rib_points)
        #ribs.addObject('PointCollisionModel')

        return rootNode

#
# AirwaySimulationTest
#


class AirwaySimulationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_AirwaySimulation1()

    def test_AirwaySimulation1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("AirwaySimulation1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = AirwaySimulationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")



def config(rootNode):
    confignode = rootNode.addChild("Config")
    confignode.addObject('RequiredPlugin', name="Sofa.Component.LinearSolver.Direct", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.SolidMechanics.FEM.Elastic", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.AnimationLoop", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Collision.Detection.Algorithm", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Collision.Detection.Intersection", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Collision.Geometry", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Collision.Response.Contact", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Lagrangian.Correction", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Lagrangian.Solver", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.IO.Mesh", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.LinearSolver.Iterative", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Mapping.NonLinear", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Mass", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.ODESolver.Backward", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.StateContainer", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Topology.Container.Constant", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.Component.Visual", printLog=False)
    confignode.addObject('RequiredPlugin', name="Sofa.GL.Component.Rendering3D", printLog=False)
    confignode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")
    confignode.addObject('RequiredPlugin', name="Sofa.Component.MechanicalLoad", printLog=False)

    rootNode.addObject('DefaultAnimationLoop')