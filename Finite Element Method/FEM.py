import numpy as np
import matplotlib.pyplot as plt


def constructNodeMatrix(nColumns: int, nRows: int, cSpace: int|float, rSpace: int|float) -> np.ndarray:
    # Creating X and Y coordinates
    xCoords = np.arange(nColumns) * cSpace
    yCoords = np.arange(nRows) * rSpace
    # Creating a grid from the coordinates
    X, Y = np.meshgrid(xCoords, yCoords) 
    # Flattening grid points
    X = X.flatten()
    Y = Y.flatten()
    # Combine flattened grid points into a matrix
    nodeMatrix = np.column_stack((X, Y))
    return nodeMatrix

def constructElementMatrix(nodeMatrix: np.ndarray, nColumns:int, nRows:int, cSpace: int|float, rSpace: int|float, youngsMod: float, crossArea:float, density: float, horizontal: bool, vertical: bool) -> np.ndarray:
    # Finding maximum X and Y coordinate (for connection logic)
    xMax, yMax = (nColumns-1)*cSpace, (nRows-1)*rSpace
    
    # Finding the angle of the diagonals (whole degrees)
    diagAngle = round(np.degrees(np.arctan(rSpace/cSpace)).item())

    elementStorage = []    
    # Looping over all nodes in the nodeMatrix
    for i in range(nodeMatrix.shape[0]):
        # The logic is build to only connect nodes from left->right and top->bottom
        x, y = nodeMatrix[i]
        
        # We only add horizontal elements if variable 'horizontal' is True
        # If we are not at right edge of the grid, we create a horizontal element
        if horizontal is True and x != xMax:
            j = i + 1  # Connected to node:
            length = cSpace
            newElement = [i, j, 0, length, crossArea, youngsMod, density]
            elementStorage.append(newElement)
        
        # We only add vertical elements if variable 'vertical' is True
        # If we are not at the top edge of the grid, we create a vertical element
        if vertical is True and y != yMax:
            j = i + nColumns # Connected to node:
            length = rSpace
            newElement = [i, j, 90, length, crossArea, youngsMod, density]
            elementStorage.append(newElement)
        
        # We always add diagonal elements where possible
        # If we are are not at the top nor right edge, we can add a diagonal element from bottom left to top right.
        if x != xMax and y != yMax:
            j = i + nColumns + 1  # Connected to node:
            length = np.sqrt(np.square(rSpace) + np.square(cSpace))
            newElement = [i, j, diagAngle, length, crossArea, youngsMod, density]
            elementStorage.append(newElement)

        # If we are not at the bottom nor right edge, we can add a diagnola element from top left to bottom right
        if x != 0 and y != yMax:
            j = i + nColumns - 1  # Connected to node:
            length = np.sqrt(np.square(rSpace) + np.square(cSpace))

            # Switch the connection indices (to adhere to the convention left->right, bottom->top)
            newElement = [j, i, -diagAngle, length, crossArea, youngsMod, density]
            elementStorage.append(newElement)
    
    # Build an elementMatrix from all created elements
    elementMatrix = np.array(elementStorage, dtype=np.float32)
    return elementMatrix

def constructStiffnessBlock(elementAngle: float) -> np.ndarray:
    # This is a helper function.
    # It computes the unscaled stiffness matrix for an element with a particular angle.

    radAngle = np.radians(elementAngle)
    c, s = np.cos(radAngle), np.sin(radAngle)
    elementMatrix = np.array([
        [c*c,   c*s,  -c*c,  -c*s],
        [c*s,   s*s,  -c*s,  -s*s],
        [-c*c, -c*s,   c*c,   c*s],
        [-c*s, -s*s,   c*s,   s*s]
    ], dtype=np.float32)
    return elementMatrix

def constructAMatrix(elementAngle: float) -> np.ndarray:
    # This is a helper function
    # It computes the transformation matrix between global and element coordinates

    radAngle = np.radians(elementAngle)
    c, s = np.cos(radAngle), np.sin(radAngle)

    A = np.array([
        [c, s, 0, 0],
        [0, 0, c, s]    
    ], dtype=np.float32)
    return A

def constructStiffnessMatrix(nodeMatrix: np.ndarray, elementMatrix: np.ndarray, basicStiffnesses: dict[int, np.ndarray]):
    # Computing the number of DOFs (all code assumes 2D)
    nDofs = nodeMatrix.shape[0]*2
    # Initialising a global stiffness matrix
    globalStiffnessMatrix = np.zeros(shape=(nDofs, nDofs))

    # Loop over each element
    for from_, to_, theta, length, section_area, youngs_modulus, density in elementMatrix:
        # We find the indices of the nodes in the global stiffness matrix
        i, j = int(from_*2), int(to_*2) 

        # We compute the element stiffness usic the 'basicStiffnesses' and scaling them with the material/geometry properties.
        elemStiffness = (youngs_modulus*section_area / length)*basicStiffnesses[theta.item()]

        # Then we add each sub-block of the element stiffness matrix to the global stiffness matrix
        globalStiffnessMatrix[i:i+2, i:i+2] += elemStiffness[:2, :2] # K_ii
        globalStiffnessMatrix[i:i+2, j:j+2] += elemStiffness[:2, 2:] # K_ij
        globalStiffnessMatrix[j:j+2, i:i+2] += elemStiffness[2:, :2] # K_ji
        globalStiffnessMatrix[j:j+2, j:j+2] += elemStiffness[2:, 2:] # K_jj

    return globalStiffnessMatrix

def constructForceVector(nodeMatrix: np.ndarray, elementMatrix: np.ndarray, gravityVector: np.ndarray, forceBoundaryConditions: list):
    # Computing the number of DOFs (all code assumes 2D)
    nDofs = nodeMatrix.shape[0]*2

    # Initialising the force vector
    forceVector = np.zeros(shape=(nDofs,))

    # Loop over each element
    for from_, to_, theta, length, section_area, youngs_modulus, density in elementMatrix:
        # We find the indices of the nodes in the global coordinates
        i, j = int(from_*2), int(to_*2)

        # We find the weight of the element
        weight = section_area*length*density
        # Multiply this by the gravity vector to find the force on each DOF.
        force = weight*gravityVector
        
        # Assigning the force to each DOF the element is connected to
        # (for linear elements, the force is just devided by 2)
        forceVector[i:i+2] += force/2
        forceVector[j:j+2] += force/2

    # Looping over the applied forces
    for node, axis, value in forceBoundaryConditions:
        # Finding which DOF it is acting on
        dof = int(node * 2 + axis)
        # Adding the force to that DOF
        forceVector[dof] += value

    return forceVector

def solve(globalStiffnessMatrix: np.ndarray, forceVector: np.ndarray, displacementBoundaryConditions: list):
    # We make a list of all DOFs
    allDofs = np.arange(globalStiffnessMatrix.shape[0])
    # We find the constrained and unconstrained DOFs (due to displacement boundary conditions)
    constrainedDofs = np.array([node*2 + axis for node, axis, _ in displacementBoundaryConditions])
    unconstrainedDofs = np.setdiff1d(allDofs, constrainedDofs)

    # Now we decompose the total equilibrium equation into 2 parts.
    # 1. u: Unconstrained DOFs
    # 2. c: Constrained DOFs

    # [K_uu K_uc] [U_u] = [F_u]
    # [K_cu K_cc] [U_c] = [F_c]

    # From this we find the equation that solves the unconstrained displacements (U_u)
    # 1. K_uu U_u = F_u - K_uc U_c
    # All other terms are already known, and created in this piece of code
    U_c = np.array([val for _, _, val in displacementBoundaryConditions])
    F_u = forceVector[unconstrainedDofs]

    K_uu = globalStiffnessMatrix[np.ix_(unconstrainedDofs, unconstrainedDofs)]
    K_uc = globalStiffnessMatrix[np.ix_(unconstrainedDofs, constrainedDofs)]

    # Now we solve for the unconstrained displacements
    RHS_1 = F_u - (K_uc @ U_c)
    U_u = np.linalg.solve(K_uu, RHS_1)
    
    # Here we combine the unconstrained and constrained displacements into one displacement vector
    U = np.zeros(globalStiffnessMatrix.shape[0])
    U[unconstrainedDofs] = U_u
    U[constrainedDofs] = U_c
    
    # Then we return the total displacement vector
    return U

def constructStrainVector(elementMatrix: np.ndarray, displacementVector: np.ndarray, basicAMatrices: dict[int, np.ndarray]):
    # Initialize a strain vector
    strainVector = np.zeros(elementMatrix.shape[0])

    for k, element in enumerate(elementMatrix):
        # Unpacking the element information
        from_, to_, theta, length, section_area, youngs_modulus, density = element
    
        # We find the indices of the nodes in the global coordinates
        i, j = int(from_*2), int(to_*2)
        
        # Constructing the interpolatoin matrix
        B = (1/length) * np.array([-1, 1])
        # Indexing the earlier pre-computed transformation matrix
        A = basicAMatrices[theta]

        # Getting the global nodal displacement from the displacement vector
        globalElemDisp = displacementVector[np.array([i, i+1, j, j+1], dtype=np.uint32)]

        # Transforming the global nodal displacement to element displacement
        localElemDisp = A @ globalElemDisp

        # Converting the element displacement to strain
        strain = B @ localElemDisp
        strainVector[k] = strain

    return strainVector

def plotResults(
        nodeMatrix: np.ndarray, elementMatrix: np.ndarray,
        displacementConditions: list, forceConditions: list,
        displacementVector: np.ndarray,
        strainVector: np.ndarray, xRange: list[float], yRange: list[float], deformationScale: float):
    fig, axis = plt.subplots(1, 3, figsize=(18, 6))
    nodalDisplacement = np.reshape(displacementVector, nodeMatrix.shape)
    maxXDisp, maxYDisp = np.max(nodalDisplacement, axis=0)
    xRange[1] += maxXDisp*deformationScale
    yRange[1] += maxYDisp*deformationScale
    for ax in axis:
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xRange)
        ax.set_ylim(yRange)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    confPlot, dispPlot, strainPlot = axis

    """ SUBPLOT 1: Configuration + Boundary Conditions """
    confPlot.set_title('Boundary Conditions Configuration')
    plotConfig = {
        'Element Color': 'gray',
        'Element Width': 2,
        'Marker Size': 40,
        'Free Marker Color': 'yellow',
        'Constrained Marker Color': 'blue',
        'Force Length': 1,
        'Force Color': 'red'}

    # Drawing Elements
    for from_, to_, *_ in elementMatrix:
        (x1, y1), (x2, y2) = nodeMatrix[int(from_)], nodeMatrix[int(to_)]
        confPlot.plot([x1, x2], [y1, y2], color=plotConfig['Element Color'], linewidth=plotConfig['Element Width'], zorder=1)
    
    # Drawing Free Nodes
    conNodes = [k for k, _, _ in displacementConditions]
    for i, (x, y) in enumerate(nodeMatrix):
        if not i in conNodes:
            confPlot.scatter(x, y, s=plotConfig['Marker Size'], c=plotConfig['Free Marker Color'], marker='o', edgecolors='black', linewidths=1)

    # Drawing constrained nodes
    for i, dof, _ in displacementConditions:
        x, y = nodeMatrix[i]
        if dof == 0: # Constrained in x
            confPlot.scatter(x, y, s=plotConfig['Marker Size'], c=plotConfig['Constrained Marker Color'], marker='>', edgecolors='black', linewidths=1)
        elif dof == 1:
            confPlot.scatter(x, y, s=plotConfig['Marker Size'], c=plotConfig['Constrained Marker Color'], marker='^', edgecolors='black', linewidths=1)
    
    # Drawing Forces
    for i, dof, _ in forceConditions:
        x, y = nodeMatrix[i]
        if dof == 0:
            confPlot.quiver(x, y, plotConfig['Force Length'], 0, units='xy', scale=1, scale_units='xy', color=plotConfig['Force Color'])
        elif dof == 1:
            confPlot.quiver(x, y, 0, plotConfig['Force Length'], units='xy', scale=1, scale_units='xy', color=plotConfig['Force Color'])

    """ SUBPLOT 2: Node Displacements """
    plotConfig = {
        'Undeformed Marker Size': 20,
        'Undeformed Marker Opacity': 0.3,
        'Deformed Marker Size': 40,
        'Color Map': 'turbo'}
    dispPlot.set_title(f'Nodal Displacement (Scale={deformationScale})')

    # Drawing orinal positions
    dispPlot.scatter(nodeMatrix[:, 0], nodeMatrix[:, 1], c='gray', s=plotConfig['Undeformed Marker Size'], alpha=plotConfig['Undeformed Marker Opacity'], zorder=1)

    # Drawing Displaced positions
    nodalDisplacement = np.reshape(displacementVector, nodeMatrix.shape)
    nodeColors = np.linalg.norm(nodalDisplacement, axis=1)
    nodesDisplaced = nodeMatrix + (nodalDisplacement * deformationScale)
    scatterDisp = dispPlot.scatter(nodesDisplaced[:, 0], nodesDisplaced[:, 1], c=nodeColors, s=plotConfig['Deformed Marker Size'], cmap=plotConfig['Color Map'], zorder=2)

    cbarDisp = fig.colorbar(scatterDisp, ax=dispPlot, fraction=0.045)
    cbarDisp.set_label('Displacement Magnitude (m)', rotation=270, labelpad=15)


    """ SUBPLOT 3: Element Strains """
    strainPlot.set_title('Element Strain')
    plotConfig = {
        'Color Map': 'turbo',
        'Element Width': 2,
        'Marker Size': 40,
        'Marker Color': 'yellow'}
    
    # Drawing all undeformed nods
    for i, (x, y) in enumerate(nodeMatrix):
        strainPlot.scatter(x, y, s=plotConfig['Marker Size'], c=plotConfig['Marker Color'], marker='o', edgecolors='black', linewidths=1, zorder=2)

    # Drawing Elements
    colorMap = plt.get_cmap(plotConfig['Color Map'])
    strainMin, strainMax = strainVector.min(), strainVector.max()
    normalize = plt.Normalize(vmin=strainMin, vmax=strainMax)

    sm = plt.cm.ScalarMappable(cmap=colorMap, norm=normalize)
    sm.set_array([])
    
    for i, (from_, to_, *_) in enumerate(elementMatrix):
        (x1, y1), (x2, y2) = nodeMatrix[int(from_)], nodeMatrix[int(to_)]
        strain = strainVector[i]
        strainPlot.plot([x1, x2], [y1, y2], color=colorMap(normalize(strain)), linewidth=plotConfig['Element Width'], zorder=1)
    cbarStrain = fig.colorbar(sm, ax=strainPlot, fraction=0.045)
    cbarStrain.set_label('Element Strain', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

def runFEM(settings: dict, forceConditions: list, displacementConditions: list, plot: bool):
    numRows, numColumns = settings['Grid']['Rows'], settings['Grid']['Columns']
    rowSpacing, colSpacing = settings['Grid']['Row Spacing'], settings['Grid']['Column Spacing']
    
    # Constructing Standard Element Matrices
    diagonalAngle = round(np.degrees(np.arctan(rowSpacing/colSpacing)).item())
    possibleAngles = [0, 90, diagonalAngle, -diagonalAngle]
    elementStiffnesses, globalToLocalMatrices = {}, {}
    for angle in possibleAngles:
        elementStiffnesses[angle] = constructStiffnessBlock(elementAngle=angle)
        globalToLocalMatrices[angle] = constructAMatrix(elementAngle=angle)
    
    # Constructing Nodal Matrix
    nodeMatrix = constructNodeMatrix(nColumns=numColumns, nRows=numRows, cSpace=colSpacing, rSpace=rowSpacing)

    # Constructing Element Matrix
    connectHorizontal = settings['Grid']['Connect Horizontal']
    connectVertical = settings['Grid']['Connect Vertical']
    elementProperties = settings['Elements']

    elementMatrix = constructElementMatrix(
        nodeMatrix=nodeMatrix, nColumns=numColumns, nRows=numRows, cSpace=colSpacing, rSpace=rowSpacing,
        youngsMod=elementProperties['Youngs Modulus'], crossArea=elementProperties['Area'], density=elementProperties['Density'],
        horizontal=connectHorizontal, vertical=connectVertical)
    
    # Constructing Global Stiffness Matrix
    stiffnessMatrix = constructStiffnessMatrix(nodeMatrix=nodeMatrix, elementMatrix=elementMatrix, basicStiffnesses=elementStiffnesses)


    # Constructing Force Vector
    gravityVector = settings['Grid']['Gravity Vector']
    forceVector = constructForceVector(nodeMatrix=nodeMatrix, elementMatrix=elementMatrix, gravityVector=gravityVector, forceBoundaryConditions=forceConditions)
    
    # Solving the FEM
    displacementVector = solve(globalStiffnessMatrix=stiffnessMatrix, forceVector=forceVector, displacementBoundaryConditions=displacementConditions)

    if plot is True:
        # Suplementary Strain Results
        strainVector = constructStrainVector(elementMatrix=elementMatrix, displacementVector=displacementVector, basicAMatrices=globalToLocalMatrices)
        xRange = [-1*colSpacing, numColumns*colSpacing]
        yRange = [-1*rowSpacing, numRows*rowSpacing]
        plotResults(nodeMatrix=nodeMatrix, elementMatrix=elementMatrix, 
            displacementConditions=displacementConditions, forceConditions=forceConditions,
            displacementVector=displacementVector, strainVector=strainVector, xRange=xRange, yRange=yRange, deformationScale=5)

    nodeDisplacements = np.reshape(displacementVector, nodeMatrix.shape)
    return nodeMatrix, elementMatrix, stiffnessMatrix, nodeDisplacements
    
def plotModes(nodeMatrix: np.ndarray, elementMatrix: np.ndarray, eigVals: np.ndarray, eigVecs: np.ndarray):
    plotConfig = {
        'Undeformed Marker Size': 20,
        'Undeformed Marker Opacity': 0.3,
        'Deformed Marker Size': 40,
        'Deformation Scale': 2,
        'Color Map': 'turbo',
        'Element Color': 'gray',
        'Element Width': 1.5}
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    axsFlat = axs.flatten()

    minEigVals = np.argpartition(eigVals, 6)[:6]
    for i, ax in enumerate(axsFlat):
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        eigenIdx = minEigVals[i]
        eigenValue = eigVals[eigenIdx]
        eigenVector = eigVecs[:, eigenIdx]
        nodeDisplacements = np.reshape(eigenVector, nodeMatrix.shape)
        
        # Plotting Nodes
        ax.set_title(f'Mode: {i+1}, Eigenvalue: {round(eigenValue.item())} (Scale={plotConfig['Deformation Scale']})')
        nodeColors = np.linalg.norm(nodeDisplacements, axis=1)
        nodesDisplaced = nodeMatrix + (nodeDisplacements * plotConfig['Deformation Scale'])
        scaterDisp = ax.scatter(nodesDisplaced[:, 0], nodesDisplaced[:, 1], c=nodeColors, s=plotConfig['Deformed Marker Size'], cmap=plotConfig['Color Map'], zorder=2)

        # Drawing deformed elements
        for from_, to_, *_ in elementMatrix:
            (x1, y1), (x2, y2) = nodesDisplaced[int(from_)], nodesDisplaced[int(to_)]
            ax.plot([x1, x2], [y1, y2], color=plotConfig['Element Color'], linewidth=plotConfig['Element Width'], zorder=1)

        cbarDisp = fig.colorbar(scaterDisp, ax=ax, fraction=0.045)
        cbarDisp.set_label('Displacement Magnitude', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.show()

def plot1DBar(nodeMatrix: np.ndarray, displacementVector: np.ndarray):
    fig, ax = plt.subplots()
    nodalDisplacement = np.reshape(displacementVector, nodeMatrix.shape)
    nodeColors = np.linalg.norm(nodalDisplacement, axis=1)

    ax.set_title(f'Nodal Displacement ({len(nodeMatrix)-1} Elements)')
    displacedNodes = nodeMatrix + nodalDisplacement
    scatterDisp = ax.scatter(displacedNodes[:, 0], displacedNodes[:, 1], c=nodeColors, cmap='turbo')
    cBar = fig.colorbar(scatterDisp, ax=ax)
    cBar.set_label('Displacement Magnitude (m)', rotation=270, labelpad=15)
    plt.show()


""" FEM Q2 """
settings = {
    'Elements': {
        'Youngs Modulus': 1e9,  # Pa
        'Area': 1e-4,  # M^2
        'Density': 1000  # Kg/m^3
    },
    'Grid': {
        'Rows': 1,
        'Columns': 51,
        'Row Spacing': 0.1,
        'Column Spacing': 1,
        'Gravity Vector': np.array([0, 0]),
        'Connect Horizontal': True,
        'Connect Vertical': False
    }
}
settings['Grid']['Column Spacing'] = 1 / (settings['Grid']['Columns'] - 1)  # Change element length, so they add up to 1
forceConditions = [[settings['Grid']['Columns']-1, 0, 1]]  # Add force in x direction on tip
dispConditions = [[i, 1, 0] for i in range(settings['Grid']['Columns'])]  # Add a y-constraint on every node (this is a consequence of using the same script for 2/3D)
dispConditions.append([0, 0, 0])  # Add x-constraint to the left most node

nodeMatrix, elementMatrix, stiffnessMatrix, nodeDisplacement = runFEM(settings=settings, forceConditions=forceConditions, displacementConditions=dispConditions, plot=False)
plot1DBar(nodeMatrix=nodeMatrix, displacementVector=nodeDisplacement)

U = nodeDisplacement[-1, 0]
print(f'Number of nodes: {settings['Grid']['Columns']} Tip Displacement: {U}')

""" FEM Q3"""
settings = {
    'Elements': {
        'Youngs Modulus': 1e9,  # Pa
        'Area': 1e-4,  # M^2
        'Density': 1000  # Kg/m^3
    },
    'Grid': {
        'Rows': 10,
        'Columns': 10,
        'Row Spacing': 1,
        'Column Spacing': 1,
        'Gravity Vector': np.array([0, -9.81]),
        'Connect Horizontal': True,
        'Connect Vertical': True
    }
}

topNodes = [i + ((settings['Grid']['Rows'] - 1) * settings['Grid']['Columns']) for i in range(settings['Grid']['Columns'])]
rightNodes = [i * settings['Grid']['Columns']  + (settings['Grid']['Columns'] - 1) for i in range(settings['Grid']['Rows'])]
allForceConditions = {
    'Top Left Y': [[99, 1, 10000]],
    'Top Complete Y': [[node, 1, 1000] for node in topNodes],
    'Top Complete X': [[node, 0, 1000] for node in topNodes],
    'Right Complete X': [[node, 0, 1000] for node in rightNodes],
    'Right Complete Y': [[node, 1, 1000] for node in rightNodes]
}

bottomNodes = [i for i in range(settings['Grid']['Columns'])]
leftNodes = [i*settings['Grid']['Columns'] for i in range(settings['Grid']['Rows'])]
allDispConditions = {
    'Fixed Origin X': [[0, 0, 0]],
    'Fixed Origin Y': [[0, 1, 0]],
    'Fixed Left X': [[node, 0, 0] for node in leftNodes],
    'Fixed Bottom Y': [[node, 1, 0] for node in bottomNodes]
}

structureLength = settings['Grid']['Rows'] * settings['Grid']['Row Spacing']
structureWidth = settings['Grid']['Columns'] * settings['Grid']['Column Spacing']
structureDepth = np.sqrt(((settings['Elements']['Area']*4) / np.pi)).item()
topArea = structureDepth*structureWidth
sideArea = structureLength*structureDepth

# Q3.1
forceConditions = []
forceConditions.extend(allForceConditions['Top Left Y'])

dispConditions = []
dispConditions.extend(allDispConditions['Fixed Left X'])
dispConditions.extend(allDispConditions['Fixed Bottom Y'])

nodeMatrix, elementMatrix, stiffnessMatrix, nodeDisplacement = runFEM(settings=settings, forceConditions=forceConditions, displacementConditions=dispConditions, plot=True)
Ux, Uy = nodeDisplacement[-1, :]
print('Question 3.1')
print(f'Displacement Of Top Left Element: X {Ux}, Y: {Uy} \n')

# Q3.2 a/b
forceConditions = []
forceConditions.extend(allForceConditions['Top Complete Y'])

dispConditions = []
dispConditions.extend(allDispConditions['Fixed Left X'])
dispConditions.extend(allDispConditions['Fixed Bottom Y'])

nodeMatrix, elementMatrix, stiffnessMatrix, nodeDisplacement = runFEM(settings=settings, forceConditions=forceConditions, displacementConditions=dispConditions, plot=True)

# Youngs Modulus Computation
topYDisplacements = [nodeDisplacement[node, 1].item() for node in topNodes]  # Y displacements of top nodes
deltaLength = sum(topYDisplacements) / len(topYDisplacements)
structureStrain = deltaLength / structureLength

totalForce = sum(f for _, _, f in forceConditions)
structureStress = totalForce / topArea

computedYoungs = structureStress / structureStrain
print('Question 3.2a')
print(f'Computed Youngs modulus: {computedYoungs} \n')

# Poissons Ratio Computation
topXDisplacements = [nodeDisplacement[node, 0].item() for node in topNodes]
deltaWidth = sum(topXDisplacements) / len(topXDisplacements)

poissonsRatio = -deltaWidth / deltaLength
print('Question 3.2b')
print(f'Computed Poisson Ratio: {poissonsRatio} \n')

# Q3.2 c
forceConditions = []
forceConditions.extend(allForceConditions['Top Complete X'])

dispConditions = []
dispConditions.extend(allDispConditions['Fixed Bottom Y'])
dispConditions.extend(allDispConditions['Fixed Origin X'])

nodeMatrix, elementMatrix, stiffnessMatrix, nodeDisplacement = runFEM(settings=settings, forceConditions=forceConditions, displacementConditions=dispConditions, plot=True)

# Shear Modulus Computation
totalForce = sum(f for _, _, f in forceConditions)
structureStress = totalForce / topArea

topXDisplacements = [nodeDisplacement[node, 0].item() for node in topNodes]
deltaWidth = sum(topXDisplacements) / len(topXDisplacements)
structureStrain = deltaWidth / structureLength

computedShearMod = structureStress / structureStrain
print('Question 3.2c')
print(f'Computed Shear Modulus: {computedShearMod} \n')


# Q3.3
settings['Grid']['Connect Horizontal'] = False  # Remove horizontal elements


# Q 3.3a
dispConditions = []
dispConditions.extend(allDispConditions['Fixed Bottom Y'])
dispConditions.extend(allDispConditions['Fixed Left X'])

forceConditions = []
forceConditions.extend(allForceConditions['Top Complete Y'])

nodeMatrix, elementMatrix, stiffnessMatrix, nodeDisplacement = runFEM(settings=settings, forceConditions=forceConditions, displacementConditions=dispConditions, plot=True)

topYDisplacements = [nodeDisplacement[node, 1].item() for node in topNodes] 
deltaLength = sum(topYDisplacements) / len(topYDisplacements)
structureStrain = deltaLength / structureLength

totalForce = sum(f for _, _, f in forceConditions)
structureStress = totalForce / topArea

computedYoungs = structureStress / structureStrain
print('Question 3.3a')
print(f'Computed Vertical Youngs modulus: {computedYoungs} \n')


# Q 3.3b
dispConditions = []
dispConditions.extend(allDispConditions['Fixed Bottom Y'])
dispConditions.extend(allDispConditions['Fixed Left X'])

forceConditions = []
forceConditions.extend(allForceConditions['Right Complete X'])

nodeMatrix, elementMatrix, stiffnessMatrix, nodeDisplacement = runFEM(settings=settings, forceConditions=forceConditions, displacementConditions=dispConditions, plot=True)

rightXDisplacements = [nodeDisplacement[node, 0].item() for node in rightNodes] 
deltaWidth = sum(rightXDisplacements) / len(rightXDisplacements)
structureStrain = deltaWidth / structureWidth

totalForce = sum(f for _, _, f in forceConditions)
structureStress = totalForce / sideArea

computedYoungs = structureStress / structureStrain
print('Question 3.3b')
print(f'Computed Horizontal Youngs modulus: {computedYoungs} \n')


# Q 3.3c
dispConditions = []
dispConditions.extend(allDispConditions['Fixed Bottom Y'])
dispConditions.extend(allDispConditions['Fixed Origin X'])

forceConditions = []
forceConditions.extend(allForceConditions['Top Complete X'])

nodeMatrix, elementMatrix, stiffnessMatrix, nodeDisplacement = runFEM(settings=settings, forceConditions=forceConditions, displacementConditions=dispConditions, plot=True)

topXDisplacements = [nodeDisplacement[node, 0].item() for node in topNodes] 
deltaWidth = sum(topXDisplacements) / len(topXDisplacements)
structureStrain = deltaWidth / structureLength

totalForce = sum(f for _, _, f in forceConditions)
structureStress = totalForce / topArea

computedShearMod = structureStress / structureStrain
print('Question 3.3c')
print(f'Computed horizontal shear modulus: {computedShearMod} \n')



# Q 3.3d
dispConditions = []
dispConditions.extend(allDispConditions['Fixed Origin Y'])
dispConditions.extend(allDispConditions['Fixed Left X'])

forceConditions = []
forceConditions.extend(allForceConditions['Right Complete Y'])

nodeMatrix, elementMatrix, stiffnessMatrix, nodeDisplacement = runFEM(settings=settings, forceConditions=forceConditions, displacementConditions=dispConditions, plot=True)

rightYDisplacements = [nodeDisplacement[node, 1].item() for node in rightNodes] 
deltaLength = sum(rightYDisplacements) / len(rightYDisplacements)
structureStrain = deltaLength / structureWidth

totalForce = sum(f for _, _, f in forceConditions)
structureStress = totalForce / sideArea

computedShearMod = structureStress / structureStrain
print('Question 3.3d')
print(f'Computed vertical shear modulus: {computedShearMod} \n')
