# ias

## Summary


## Installation

To install ias, you will need to install the following packages first:  
* OpenMPI: Tools for MPI including compiler. To install this package, you can either install it via a package manager, e.g. in Ubuntu:
```
sudo apt install libopenmpi-dev
```
or directly from the source code (https://www.open-mpi.org/software/ompi/v4.1/), e.g.
```
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz
tar -zxvf openmpi-4.1.1.tar.gz
cd openmpi-4.1.1
./configure --prefix=MPI_PATH
make all
make install
```
where you have to change ```MPI_PATH``` to the path where you want OpenMPI to be installed.
* Trilinos: This package is used to handle linear algebra objects and solvers in a distributed memory fashion (MPI). To install this package, you can either install it via a package manager, e.g. in Ubuntu:
```
sudo apt install trilinos-all-dev
```
or directly from the source code (https://github.com/trilinos/trilinos). With git, follow these steps
```
git clone https://github.com/trilinos/Trilinos.git
```
will download the project in a new folder called "Trilinos". Inside that folder, create a build folder, configure and make:
```
cd Trilinos
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS:BOOL=YES \
      -DTrilinos_ENABLE_Amesos2:BOOL=YES \
      -DTrilinos_ENABLE_AztecOO:BOOL=YES \
      -DTrilinos_ENABLE_Amesos:BOOL=YES \
      -DTrilinos_ENABLE_Belos:BOOL=YES \
      -DTrilinos_ENABLE_Epetra:BOOL=YES \
      -DTrilinos_ENABLE_Isorropia:BOOL=YES \
      -DTrilinos_ENABLE_OpenMP=ON \
      -DTrilinos_ENABLE_Fortran=YES \
      -DTPL_ENABLE_MPI:BOOL=YES \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DTPL_ENABLE_Netcdf=OFF \
      -DTrilinos_ENABLE_PyTrilinos=OFF \
      -DTrilinos_ENABLE_STK:BOOL=OFF \
      -DCMAKE_C_FLAGS="-O3" \
      -DCMAKE_CXX_FLAGS="-O3" \
      -DCMAKE_FORTRAN_FLAGS="-O3" \
      -DCMAKE_C_COMPILER="MPI_PATH/bin/mpicc" \
      -DCMAKE_CXX_COMPILER="MPI_PATH/bin/mpicxx" \
      -DCMAKE_FORTRAN_COMPILER="MPI_PATH/bin/mpif90" \
      -DCMAKE_INSTALL_PREFIX=TRILINOS_PATH \
      ..
make install
```
where you have to change ```TRILINOS_PATH``` to the path where you want Trilinos to be installed and ```MPI_PATH``` to the path to where MPI was installed.
* VTK: This library is used for IO. Again you can directly install it with your package manager, e.g. in Ubuntu
```
sudo apt install vtk9
```
or you can install it from sources  (https://vtk.org/download/), e.g.
```
wget https://www.vtk.org/files/release/9.0/VTK-9.0.1.tar.gz
tar -zxvf VTK-9.0.1
cd VTK-9.0.1
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS:BOOL=YES \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER="MPI_PATH/bin/mpicc" \
      -DCMAKE_CXX_COMPILER="MPI_PATH/bin/mpicxx" \
      -DCMAKE_INSTALL_PREFIX=VTK_PATH \
      ..
```
where you have to change ```VTK_PATH``` to the path where you want VTK to be installed and ```MPI_PATH``` to the path to where MPI was installed (if you don't know it, try typing ```which mpicc```, which should give you ```MPI_PATH/bin```). 

Once all previous packages all installed, then to install ias. First download the code if you don't have it already
```
git clone git@github.com:torressancheza/ias.git
```
Inside the ias folder,
```
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS:BOOL=YES \
      -DTrilinos_DIR=TRILINOS_PATH/lib/cmake/Trilinos \
      -DVTK_DIR=VTK_PATH/lib/cmake/vtk-X \
      -DCMAKE_CXX_COMPILER=CXX \
      -DCMAKE_C_COMPILER=CC \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=IAS_PATH \
      ..
make install
```
where ```IAS_PATH```is the path where you want ias to be installed, ```TRILINOS_PATH``` is the path where Trilinos was installed, ```VTK_PATH``` the path where VTK was installed (note that in the same line ```X``` should match the Vtk version, e.g. ```9.0```), and ```MPI_PATH``` the path to where MPI was installed. If you are working on MacOS and want to use Xcode, you might want to add right after ```..```, ```-G Xcode``` so that Xcode files are generated.
## Usage

## Authors
[Alejandro Torres-Sánchez](https://torres-sanchez.xyz/)

Max Kerr Winter

Guillaume Salbreux

## Licence
The code is provided under the MIT licence (see licence.txt).
