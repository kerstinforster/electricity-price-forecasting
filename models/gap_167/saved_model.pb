??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
z
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_36/kernel
s
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel*
_output_shapes

:@*
dtype0
r
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_36/bias
k
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes
:@*
dtype0
z
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@* 
shared_namedense_37/kernel
s
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes

:@@*
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
_output_shapes
:@*
dtype0
{
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?T* 
shared_namedense_38/kernel
t
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel*
_output_shapes
:	?T*
dtype0
r
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_38/bias
k
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_36/kernel/m
?
*Adam/dense_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/m*
_output_shapes

:@*
dtype0
?
Adam/dense_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_36/bias/m
y
(Adam/dense_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_37/kernel/m
?
*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m*
_output_shapes

:@@*
dtype0
?
Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_37/bias/m
y
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?T*'
shared_nameAdam/dense_38/kernel/m
?
*Adam/dense_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/m*
_output_shapes
:	?T*
dtype0
?
Adam/dense_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_38/bias/m
y
(Adam/dense_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_36/kernel/v
?
*Adam/dense_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/v*
_output_shapes

:@*
dtype0
?
Adam/dense_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_36/bias/v
y
(Adam/dense_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*'
shared_nameAdam/dense_37/kernel/v
?
*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v*
_output_shapes

:@@*
dtype0
?
Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_37/bias/v
y
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?T*'
shared_nameAdam/dense_38/kernel/v
?
*Adam/dense_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/kernel/v*
_output_shapes
:	?T*
dtype0
?
Adam/dense_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_38/bias/v
y
(Adam/dense_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_38/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?)
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?)
value?)B?) B?)
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
h

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
?
&iter

'beta_1

(beta_2
	)decay
*learning_ratemTmUmVmW mX!mYvZv[v\v] v^!v_
*
0
1
2
3
 4
!5
 
*
0
1
2
3
 4
!5
?
+metrics
,layer_metrics
	variables

-layers
.layer_regularization_losses
/non_trainable_variables
regularization_losses
	trainable_variables
 
[Y
VARIABLE_VALUEdense_36/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_36/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
0metrics
1layer_metrics
	variables

2layers
3layer_regularization_losses
4non_trainable_variables
regularization_losses
trainable_variables
[Y
VARIABLE_VALUEdense_37/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_37/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
5metrics
6layer_metrics
	variables

7layers
8layer_regularization_losses
9non_trainable_variables
regularization_losses
trainable_variables
 
 
 
?
:metrics
;layer_metrics
	variables

<layers
=layer_regularization_losses
>non_trainable_variables
regularization_losses
trainable_variables
 
 
 
?
?metrics
@layer_metrics
	variables

Alayers
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses
trainable_variables
[Y
VARIABLE_VALUEdense_38/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_38/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
?
Dmetrics
Elayer_metrics
"	variables

Flayers
Glayer_regularization_losses
Hnon_trainable_variables
#regularization_losses
$trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

I0
J1
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ktotal
	Lcount
M	variables
N	keras_api
D
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

M	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1

R	variables
~|
VARIABLE_VALUEAdam/dense_36/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_36/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_38/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_38/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_36/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_36/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_38/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_38/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_dense_36_inputPlaceholder*,
_output_shapes
:??????????*
dtype0*!
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_36_inputdense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_1949830
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOp#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_36/kernel/m/Read/ReadVariableOp(Adam/dense_36/bias/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOp*Adam/dense_38/kernel/m/Read/ReadVariableOp(Adam/dense_38/bias/m/Read/ReadVariableOp*Adam/dense_36/kernel/v/Read/ReadVariableOp(Adam/dense_36/bias/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOp*Adam/dense_38/kernel/v/Read/ReadVariableOp(Adam/dense_38/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_1950406
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_36/kerneldense_36/biasdense_37/kerneldense_37/biasdense_38/kerneldense_38/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_36/kernel/mAdam/dense_36/bias/mAdam/dense_37/kernel/mAdam/dense_37/bias/mAdam/dense_38/kernel/mAdam/dense_38/bias/mAdam/dense_36/kernel/vAdam/dense_36/bias/vAdam/dense_37/kernel/vAdam/dense_37/bias/vAdam/dense_38/kernel/vAdam/dense_38/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_1950497??

?9
?
E__inference_dense_36_layer_call_and_return_conditional_losses_1949377

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_36/kernel/Regularizer/Abs/ReadVariableOp?1dense_36/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
Relu?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1r
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_36/kernel/Regularizer/Abs/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_36/kernel/Regularizer/Abs/ReadVariableOp.dense_36/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
/__inference_sequential_24_layer_call_fn_1949673
dense_36_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:	?T
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_24_layer_call_and_return_conditional_losses_19496412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:??????????
(
_user_specified_namedense_36_input
?
?
__inference_loss_fn_1_1950302I
7dense_37_kernel_regularizer_abs_readvariableop_resource:@@
identity??.dense_37/kernel/Regularizer/Abs/ReadVariableOp?1dense_37/kernel/Regularizer/Square/ReadVariableOp?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_37_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@@*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_37_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1o
IdentityIdentity%dense_37/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp/^dense_37/kernel/Regularizer/Abs/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_37/kernel/Regularizer/Abs/ReadVariableOp.dense_37/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp
?

?
E__inference_dense_38_layer_call_and_return_conditional_losses_1949460

inputs1
matmul_readvariableop_resource:	?T-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????T
 
_user_specified_nameinputs
?	
?
%__inference_signature_wrapper_1949830
dense_36_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:	?T
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_19493242
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:??????????
(
_user_specified_namedense_36_input
?
d
F__inference_dropout_6_layer_call_and_return_conditional_losses_1949440

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_1950282I
7dense_36_kernel_regularizer_abs_readvariableop_resource:@
identity??.dense_36/kernel/Regularizer/Abs/ReadVariableOp?1dense_36/kernel/Regularizer/Square/ReadVariableOp?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7dense_36_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7dense_36_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1o
IdentityIdentity%dense_36/kernel/Regularizer/add_1:z:0^NoOp*
T0*
_output_shapes
: 2

Identity?
NoOpNoOp/^dense_36/kernel/Regularizer/Abs/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.dense_36/kernel/Regularizer/Abs/ReadVariableOp.dense_36/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp
?
c
G__inference_flatten_24_layer_call_and_return_conditional_losses_1949448

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? *  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????T2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
??
?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1950031

inputs<
*dense_36_tensordot_readvariableop_resource:@6
(dense_36_biasadd_readvariableop_resource:@<
*dense_37_tensordot_readvariableop_resource:@@6
(dense_37_biasadd_readvariableop_resource:@:
'dense_38_matmul_readvariableop_resource:	?T6
(dense_38_biasadd_readvariableop_resource:
identity??dense_36/BiasAdd/ReadVariableOp?!dense_36/Tensordot/ReadVariableOp?.dense_36/kernel/Regularizer/Abs/ReadVariableOp?1dense_36/kernel/Regularizer/Square/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?!dense_37/Tensordot/ReadVariableOp?.dense_37/kernel/Regularizer/Abs/ReadVariableOp?1dense_37/kernel/Regularizer/Square/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?
!dense_36/Tensordot/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02#
!dense_36/Tensordot/ReadVariableOp|
dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_36/Tensordot/axes?
dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_36/Tensordot/freej
dense_36/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_36/Tensordot/Shape?
 dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_36/Tensordot/GatherV2/axis?
dense_36/Tensordot/GatherV2GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/free:output:0)dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_36/Tensordot/GatherV2?
"dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_36/Tensordot/GatherV2_1/axis?
dense_36/Tensordot/GatherV2_1GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/axes:output:0+dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_36/Tensordot/GatherV2_1~
dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const?
dense_36/Tensordot/ProdProd$dense_36/Tensordot/GatherV2:output:0!dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_36/Tensordot/Prod?
dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const_1?
dense_36/Tensordot/Prod_1Prod&dense_36/Tensordot/GatherV2_1:output:0#dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/Tensordot/Prod_1?
dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_36/Tensordot/concat/axis?
dense_36/Tensordot/concatConcatV2 dense_36/Tensordot/free:output:0 dense_36/Tensordot/axes:output:0'dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/concat?
dense_36/Tensordot/stackPack dense_36/Tensordot/Prod:output:0"dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/stack?
dense_36/Tensordot/transpose	Transposeinputs"dense_36/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_36/Tensordot/transpose?
dense_36/Tensordot/ReshapeReshape dense_36/Tensordot/transpose:y:0!dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_36/Tensordot/Reshape?
dense_36/Tensordot/MatMulMatMul#dense_36/Tensordot/Reshape:output:0)dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_36/Tensordot/MatMul?
dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_36/Tensordot/Const_2?
 dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_36/Tensordot/concat_1/axis?
dense_36/Tensordot/concat_1ConcatV2$dense_36/Tensordot/GatherV2:output:0#dense_36/Tensordot/Const_2:output:0)dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/concat_1?
dense_36/TensordotReshape#dense_36/Tensordot/MatMul:product:0$dense_36/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
dense_36/Tensordot?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/Tensordot:output:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
dense_36/BiasAddx
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
dense_36/Relu?
!dense_37/Tensordot/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02#
!dense_37/Tensordot/ReadVariableOp|
dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_37/Tensordot/axes?
dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_37/Tensordot/free
dense_37/Tensordot/ShapeShapedense_36/Relu:activations:0*
T0*
_output_shapes
:2
dense_37/Tensordot/Shape?
 dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/GatherV2/axis?
dense_37/Tensordot/GatherV2GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/free:output:0)dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2?
"dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_37/Tensordot/GatherV2_1/axis?
dense_37/Tensordot/GatherV2_1GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/axes:output:0+dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2_1~
dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const?
dense_37/Tensordot/ProdProd$dense_37/Tensordot/GatherV2:output:0!dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod?
dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const_1?
dense_37/Tensordot/Prod_1Prod&dense_37/Tensordot/GatherV2_1:output:0#dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod_1?
dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_37/Tensordot/concat/axis?
dense_37/Tensordot/concatConcatV2 dense_37/Tensordot/free:output:0 dense_37/Tensordot/axes:output:0'dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat?
dense_37/Tensordot/stackPack dense_37/Tensordot/Prod:output:0"dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/stack?
dense_37/Tensordot/transpose	Transposedense_36/Relu:activations:0"dense_37/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@2
dense_37/Tensordot/transpose?
dense_37/Tensordot/ReshapeReshape dense_37/Tensordot/transpose:y:0!dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_37/Tensordot/Reshape?
dense_37/Tensordot/MatMulMatMul#dense_37/Tensordot/Reshape:output:0)dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_37/Tensordot/MatMul?
dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_37/Tensordot/Const_2?
 dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/concat_1/axis?
dense_37/Tensordot/concat_1ConcatV2$dense_37/Tensordot/GatherV2:output:0#dense_37/Tensordot/Const_2:output:0)dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat_1?
dense_37/TensordotReshape#dense_37/Tensordot/MatMul:product:0$dense_37/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
dense_37/Tensordot?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/Tensordot:output:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
dense_37/BiasAddx
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
dense_37/Reluw
dropout_6/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout_6/dropout/Const?
dropout_6/dropout/MulMuldense_37/Relu:activations:0 dropout_6/dropout/Const:output:0*
T0*,
_output_shapes
:??????????@2
dropout_6/dropout/Mul}
dropout_6/dropout/ShapeShapedense_37/Relu:activations:0*
T0*
_output_shapes
:2
dropout_6/dropout/Shape?
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype020
.dropout_6/dropout/random_uniform/RandomUniform?
 dropout_6/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2"
 dropout_6/dropout/GreaterEqual/y?
dropout_6/dropout/GreaterEqualGreaterEqual7dropout_6/dropout/random_uniform/RandomUniform:output:0)dropout_6/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@2 
dropout_6/dropout/GreaterEqual?
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@2
dropout_6/dropout/Cast?
dropout_6/dropout/Mul_1Muldropout_6/dropout/Mul:z:0dropout_6/dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@2
dropout_6/dropout/Mul_1u
flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? *  2
flatten_24/Const?
flatten_24/ReshapeReshapedropout_6/dropout/Mul_1:z:0flatten_24/Const:output:0*
T0*(
_output_shapes
:??????????T2
flatten_24/Reshape?
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes
:	?T*
dtype02 
dense_38/MatMul/ReadVariableOp?
dense_38/MatMulMatMulflatten_24/Reshape:output:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_38/MatMul?
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_38/BiasAdd/ReadVariableOp?
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_38/BiasAdd?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1t
IdentityIdentitydense_38/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_36/BiasAdd/ReadVariableOp"^dense_36/Tensordot/ReadVariableOp/^dense_36/kernel/Regularizer/Abs/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp"^dense_37/Tensordot/ReadVariableOp/^dense_37/kernel/Regularizer/Abs/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2F
!dense_36/Tensordot/ReadVariableOp!dense_36/Tensordot/ReadVariableOp2`
.dense_36/kernel/Regularizer/Abs/ReadVariableOp.dense_36/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2F
!dense_37/Tensordot/ReadVariableOp!dense_37/Tensordot/ReadVariableOp2`
.dense_37/kernel/Regularizer/Abs/ReadVariableOp.dense_37/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
H
,__inference_flatten_24_layer_call_fn_1950243

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_24_layer_call_and_return_conditional_losses_19494482
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
*__inference_dense_36_layer_call_fn_1950135

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_36_layer_call_and_return_conditional_losses_19493772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?9
?
E__inference_dense_37_layer_call_and_return_conditional_losses_1950196

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_37/kernel/Regularizer/Abs/ReadVariableOp?1dense_37/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
Relu?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1r
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_37/kernel/Regularizer/Abs/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_37/kernel/Regularizer/Abs/ReadVariableOp.dense_37/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?	
?
/__inference_sequential_24_layer_call_fn_1950048

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:	?T
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_24_layer_call_and_return_conditional_losses_19494972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_6_layer_call_and_return_conditional_losses_1950222

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
*__inference_dense_37_layer_call_fn_1950205

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_37_layer_call_and_return_conditional_losses_19494292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?o
?
"__inference__wrapped_model_1949324
dense_36_inputJ
8sequential_24_dense_36_tensordot_readvariableop_resource:@D
6sequential_24_dense_36_biasadd_readvariableop_resource:@J
8sequential_24_dense_37_tensordot_readvariableop_resource:@@D
6sequential_24_dense_37_biasadd_readvariableop_resource:@H
5sequential_24_dense_38_matmul_readvariableop_resource:	?TD
6sequential_24_dense_38_biasadd_readvariableop_resource:
identity??-sequential_24/dense_36/BiasAdd/ReadVariableOp?/sequential_24/dense_36/Tensordot/ReadVariableOp?-sequential_24/dense_37/BiasAdd/ReadVariableOp?/sequential_24/dense_37/Tensordot/ReadVariableOp?-sequential_24/dense_38/BiasAdd/ReadVariableOp?,sequential_24/dense_38/MatMul/ReadVariableOp?
/sequential_24/dense_36/Tensordot/ReadVariableOpReadVariableOp8sequential_24_dense_36_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype021
/sequential_24/dense_36/Tensordot/ReadVariableOp?
%sequential_24/dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_24/dense_36/Tensordot/axes?
%sequential_24/dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_24/dense_36/Tensordot/free?
&sequential_24/dense_36/Tensordot/ShapeShapedense_36_input*
T0*
_output_shapes
:2(
&sequential_24/dense_36/Tensordot/Shape?
.sequential_24/dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_24/dense_36/Tensordot/GatherV2/axis?
)sequential_24/dense_36/Tensordot/GatherV2GatherV2/sequential_24/dense_36/Tensordot/Shape:output:0.sequential_24/dense_36/Tensordot/free:output:07sequential_24/dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_24/dense_36/Tensordot/GatherV2?
0sequential_24/dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_24/dense_36/Tensordot/GatherV2_1/axis?
+sequential_24/dense_36/Tensordot/GatherV2_1GatherV2/sequential_24/dense_36/Tensordot/Shape:output:0.sequential_24/dense_36/Tensordot/axes:output:09sequential_24/dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_24/dense_36/Tensordot/GatherV2_1?
&sequential_24/dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_24/dense_36/Tensordot/Const?
%sequential_24/dense_36/Tensordot/ProdProd2sequential_24/dense_36/Tensordot/GatherV2:output:0/sequential_24/dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_24/dense_36/Tensordot/Prod?
(sequential_24/dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_24/dense_36/Tensordot/Const_1?
'sequential_24/dense_36/Tensordot/Prod_1Prod4sequential_24/dense_36/Tensordot/GatherV2_1:output:01sequential_24/dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_24/dense_36/Tensordot/Prod_1?
,sequential_24/dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_24/dense_36/Tensordot/concat/axis?
'sequential_24/dense_36/Tensordot/concatConcatV2.sequential_24/dense_36/Tensordot/free:output:0.sequential_24/dense_36/Tensordot/axes:output:05sequential_24/dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_24/dense_36/Tensordot/concat?
&sequential_24/dense_36/Tensordot/stackPack.sequential_24/dense_36/Tensordot/Prod:output:00sequential_24/dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_24/dense_36/Tensordot/stack?
*sequential_24/dense_36/Tensordot/transpose	Transposedense_36_input0sequential_24/dense_36/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2,
*sequential_24/dense_36/Tensordot/transpose?
(sequential_24/dense_36/Tensordot/ReshapeReshape.sequential_24/dense_36/Tensordot/transpose:y:0/sequential_24/dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_24/dense_36/Tensordot/Reshape?
'sequential_24/dense_36/Tensordot/MatMulMatMul1sequential_24/dense_36/Tensordot/Reshape:output:07sequential_24/dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'sequential_24/dense_36/Tensordot/MatMul?
(sequential_24/dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_24/dense_36/Tensordot/Const_2?
.sequential_24/dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_24/dense_36/Tensordot/concat_1/axis?
)sequential_24/dense_36/Tensordot/concat_1ConcatV22sequential_24/dense_36/Tensordot/GatherV2:output:01sequential_24/dense_36/Tensordot/Const_2:output:07sequential_24/dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_24/dense_36/Tensordot/concat_1?
 sequential_24/dense_36/TensordotReshape1sequential_24/dense_36/Tensordot/MatMul:product:02sequential_24/dense_36/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2"
 sequential_24/dense_36/Tensordot?
-sequential_24/dense_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_36_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_24/dense_36/BiasAdd/ReadVariableOp?
sequential_24/dense_36/BiasAddBiasAdd)sequential_24/dense_36/Tensordot:output:05sequential_24/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2 
sequential_24/dense_36/BiasAdd?
sequential_24/dense_36/ReluRelu'sequential_24/dense_36/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
sequential_24/dense_36/Relu?
/sequential_24/dense_37/Tensordot/ReadVariableOpReadVariableOp8sequential_24_dense_37_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype021
/sequential_24/dense_37/Tensordot/ReadVariableOp?
%sequential_24/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_24/dense_37/Tensordot/axes?
%sequential_24/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_24/dense_37/Tensordot/free?
&sequential_24/dense_37/Tensordot/ShapeShape)sequential_24/dense_36/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_24/dense_37/Tensordot/Shape?
.sequential_24/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_24/dense_37/Tensordot/GatherV2/axis?
)sequential_24/dense_37/Tensordot/GatherV2GatherV2/sequential_24/dense_37/Tensordot/Shape:output:0.sequential_24/dense_37/Tensordot/free:output:07sequential_24/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_24/dense_37/Tensordot/GatherV2?
0sequential_24/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_24/dense_37/Tensordot/GatherV2_1/axis?
+sequential_24/dense_37/Tensordot/GatherV2_1GatherV2/sequential_24/dense_37/Tensordot/Shape:output:0.sequential_24/dense_37/Tensordot/axes:output:09sequential_24/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_24/dense_37/Tensordot/GatherV2_1?
&sequential_24/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_24/dense_37/Tensordot/Const?
%sequential_24/dense_37/Tensordot/ProdProd2sequential_24/dense_37/Tensordot/GatherV2:output:0/sequential_24/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_24/dense_37/Tensordot/Prod?
(sequential_24/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_24/dense_37/Tensordot/Const_1?
'sequential_24/dense_37/Tensordot/Prod_1Prod4sequential_24/dense_37/Tensordot/GatherV2_1:output:01sequential_24/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_24/dense_37/Tensordot/Prod_1?
,sequential_24/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_24/dense_37/Tensordot/concat/axis?
'sequential_24/dense_37/Tensordot/concatConcatV2.sequential_24/dense_37/Tensordot/free:output:0.sequential_24/dense_37/Tensordot/axes:output:05sequential_24/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_24/dense_37/Tensordot/concat?
&sequential_24/dense_37/Tensordot/stackPack.sequential_24/dense_37/Tensordot/Prod:output:00sequential_24/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_24/dense_37/Tensordot/stack?
*sequential_24/dense_37/Tensordot/transpose	Transpose)sequential_24/dense_36/Relu:activations:00sequential_24/dense_37/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@2,
*sequential_24/dense_37/Tensordot/transpose?
(sequential_24/dense_37/Tensordot/ReshapeReshape.sequential_24/dense_37/Tensordot/transpose:y:0/sequential_24/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_24/dense_37/Tensordot/Reshape?
'sequential_24/dense_37/Tensordot/MatMulMatMul1sequential_24/dense_37/Tensordot/Reshape:output:07sequential_24/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2)
'sequential_24/dense_37/Tensordot/MatMul?
(sequential_24/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2*
(sequential_24/dense_37/Tensordot/Const_2?
.sequential_24/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_24/dense_37/Tensordot/concat_1/axis?
)sequential_24/dense_37/Tensordot/concat_1ConcatV22sequential_24/dense_37/Tensordot/GatherV2:output:01sequential_24/dense_37/Tensordot/Const_2:output:07sequential_24/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_24/dense_37/Tensordot/concat_1?
 sequential_24/dense_37/TensordotReshape1sequential_24/dense_37/Tensordot/MatMul:product:02sequential_24/dense_37/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2"
 sequential_24/dense_37/Tensordot?
-sequential_24/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_24/dense_37/BiasAdd/ReadVariableOp?
sequential_24/dense_37/BiasAddBiasAdd)sequential_24/dense_37/Tensordot:output:05sequential_24/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2 
sequential_24/dense_37/BiasAdd?
sequential_24/dense_37/ReluRelu'sequential_24/dense_37/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
sequential_24/dense_37/Relu?
 sequential_24/dropout_6/IdentityIdentity)sequential_24/dense_37/Relu:activations:0*
T0*,
_output_shapes
:??????????@2"
 sequential_24/dropout_6/Identity?
sequential_24/flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? *  2 
sequential_24/flatten_24/Const?
 sequential_24/flatten_24/ReshapeReshape)sequential_24/dropout_6/Identity:output:0'sequential_24/flatten_24/Const:output:0*
T0*(
_output_shapes
:??????????T2"
 sequential_24/flatten_24/Reshape?
,sequential_24/dense_38/MatMul/ReadVariableOpReadVariableOp5sequential_24_dense_38_matmul_readvariableop_resource*
_output_shapes
:	?T*
dtype02.
,sequential_24/dense_38/MatMul/ReadVariableOp?
sequential_24/dense_38/MatMulMatMul)sequential_24/flatten_24/Reshape:output:04sequential_24/dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_24/dense_38/MatMul?
-sequential_24/dense_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_24_dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_24/dense_38/BiasAdd/ReadVariableOp?
sequential_24/dense_38/BiasAddBiasAdd'sequential_24/dense_38/MatMul:product:05sequential_24/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_24/dense_38/BiasAdd?
IdentityIdentity'sequential_24/dense_38/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp.^sequential_24/dense_36/BiasAdd/ReadVariableOp0^sequential_24/dense_36/Tensordot/ReadVariableOp.^sequential_24/dense_37/BiasAdd/ReadVariableOp0^sequential_24/dense_37/Tensordot/ReadVariableOp.^sequential_24/dense_38/BiasAdd/ReadVariableOp-^sequential_24/dense_38/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 2^
-sequential_24/dense_36/BiasAdd/ReadVariableOp-sequential_24/dense_36/BiasAdd/ReadVariableOp2b
/sequential_24/dense_36/Tensordot/ReadVariableOp/sequential_24/dense_36/Tensordot/ReadVariableOp2^
-sequential_24/dense_37/BiasAdd/ReadVariableOp-sequential_24/dense_37/BiasAdd/ReadVariableOp2b
/sequential_24/dense_37/Tensordot/ReadVariableOp/sequential_24/dense_37/Tensordot/ReadVariableOp2^
-sequential_24/dense_38/BiasAdd/ReadVariableOp-sequential_24/dense_38/BiasAdd/ReadVariableOp2\
,sequential_24/dense_38/MatMul/ReadVariableOp,sequential_24/dense_38/MatMul/ReadVariableOp:\ X
,
_output_shapes
:??????????
(
_user_specified_namedense_36_input
?
e
F__inference_dropout_6_layer_call_and_return_conditional_losses_1949548

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:??????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:??????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:??????????@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:??????????@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:??????????@2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?9
?
E__inference_dense_36_layer_call_and_return_conditional_losses_1950126

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_36/kernel/Regularizer/Abs/ReadVariableOp?1dense_36/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
Relu?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1r
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_36/kernel/Regularizer/Abs/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_36/kernel/Regularizer/Abs/ReadVariableOp.dense_36/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949927

inputs<
*dense_36_tensordot_readvariableop_resource:@6
(dense_36_biasadd_readvariableop_resource:@<
*dense_37_tensordot_readvariableop_resource:@@6
(dense_37_biasadd_readvariableop_resource:@:
'dense_38_matmul_readvariableop_resource:	?T6
(dense_38_biasadd_readvariableop_resource:
identity??dense_36/BiasAdd/ReadVariableOp?!dense_36/Tensordot/ReadVariableOp?.dense_36/kernel/Regularizer/Abs/ReadVariableOp?1dense_36/kernel/Regularizer/Square/ReadVariableOp?dense_37/BiasAdd/ReadVariableOp?!dense_37/Tensordot/ReadVariableOp?.dense_37/kernel/Regularizer/Abs/ReadVariableOp?1dense_37/kernel/Regularizer/Square/ReadVariableOp?dense_38/BiasAdd/ReadVariableOp?dense_38/MatMul/ReadVariableOp?
!dense_36/Tensordot/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02#
!dense_36/Tensordot/ReadVariableOp|
dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_36/Tensordot/axes?
dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_36/Tensordot/freej
dense_36/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_36/Tensordot/Shape?
 dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_36/Tensordot/GatherV2/axis?
dense_36/Tensordot/GatherV2GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/free:output:0)dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_36/Tensordot/GatherV2?
"dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_36/Tensordot/GatherV2_1/axis?
dense_36/Tensordot/GatherV2_1GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/axes:output:0+dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_36/Tensordot/GatherV2_1~
dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const?
dense_36/Tensordot/ProdProd$dense_36/Tensordot/GatherV2:output:0!dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_36/Tensordot/Prod?
dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const_1?
dense_36/Tensordot/Prod_1Prod&dense_36/Tensordot/GatherV2_1:output:0#dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/Tensordot/Prod_1?
dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_36/Tensordot/concat/axis?
dense_36/Tensordot/concatConcatV2 dense_36/Tensordot/free:output:0 dense_36/Tensordot/axes:output:0'dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/concat?
dense_36/Tensordot/stackPack dense_36/Tensordot/Prod:output:0"dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/stack?
dense_36/Tensordot/transpose	Transposeinputs"dense_36/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????2
dense_36/Tensordot/transpose?
dense_36/Tensordot/ReshapeReshape dense_36/Tensordot/transpose:y:0!dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_36/Tensordot/Reshape?
dense_36/Tensordot/MatMulMatMul#dense_36/Tensordot/Reshape:output:0)dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_36/Tensordot/MatMul?
dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_36/Tensordot/Const_2?
 dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_36/Tensordot/concat_1/axis?
dense_36/Tensordot/concat_1ConcatV2$dense_36/Tensordot/GatherV2:output:0#dense_36/Tensordot/Const_2:output:0)dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/concat_1?
dense_36/TensordotReshape#dense_36/Tensordot/MatMul:product:0$dense_36/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
dense_36/Tensordot?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/Tensordot:output:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
dense_36/BiasAddx
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
dense_36/Relu?
!dense_37/Tensordot/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02#
!dense_37/Tensordot/ReadVariableOp|
dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_37/Tensordot/axes?
dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_37/Tensordot/free
dense_37/Tensordot/ShapeShapedense_36/Relu:activations:0*
T0*
_output_shapes
:2
dense_37/Tensordot/Shape?
 dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/GatherV2/axis?
dense_37/Tensordot/GatherV2GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/free:output:0)dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2?
"dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_37/Tensordot/GatherV2_1/axis?
dense_37/Tensordot/GatherV2_1GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/axes:output:0+dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2_1~
dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const?
dense_37/Tensordot/ProdProd$dense_37/Tensordot/GatherV2:output:0!dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod?
dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const_1?
dense_37/Tensordot/Prod_1Prod&dense_37/Tensordot/GatherV2_1:output:0#dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod_1?
dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_37/Tensordot/concat/axis?
dense_37/Tensordot/concatConcatV2 dense_37/Tensordot/free:output:0 dense_37/Tensordot/axes:output:0'dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat?
dense_37/Tensordot/stackPack dense_37/Tensordot/Prod:output:0"dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/stack?
dense_37/Tensordot/transpose	Transposedense_36/Relu:activations:0"dense_37/Tensordot/concat:output:0*
T0*,
_output_shapes
:??????????@2
dense_37/Tensordot/transpose?
dense_37/Tensordot/ReshapeReshape dense_37/Tensordot/transpose:y:0!dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_37/Tensordot/Reshape?
dense_37/Tensordot/MatMulMatMul#dense_37/Tensordot/Reshape:output:0)dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_37/Tensordot/MatMul?
dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
dense_37/Tensordot/Const_2?
 dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/concat_1/axis?
dense_37/Tensordot/concat_1ConcatV2$dense_37/Tensordot/GatherV2:output:0#dense_37/Tensordot/Const_2:output:0)dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat_1?
dense_37/TensordotReshape#dense_37/Tensordot/MatMul:product:0$dense_37/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
dense_37/Tensordot?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/Tensordot:output:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2
dense_37/BiasAddx
dense_37/ReluReludense_37/BiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
dense_37/Relu?
dropout_6/IdentityIdentitydense_37/Relu:activations:0*
T0*,
_output_shapes
:??????????@2
dropout_6/Identityu
flatten_24/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? *  2
flatten_24/Const?
flatten_24/ReshapeReshapedropout_6/Identity:output:0flatten_24/Const:output:0*
T0*(
_output_shapes
:??????????T2
flatten_24/Reshape?
dense_38/MatMul/ReadVariableOpReadVariableOp'dense_38_matmul_readvariableop_resource*
_output_shapes
:	?T*
dtype02 
dense_38/MatMul/ReadVariableOp?
dense_38/MatMulMatMulflatten_24/Reshape:output:0&dense_38/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_38/MatMul?
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_38/BiasAdd/ReadVariableOp?
dense_38/BiasAddBiasAdddense_38/MatMul:product:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_38/BiasAdd?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1t
IdentityIdentitydense_38/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp ^dense_36/BiasAdd/ReadVariableOp"^dense_36/Tensordot/ReadVariableOp/^dense_36/kernel/Regularizer/Abs/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp ^dense_37/BiasAdd/ReadVariableOp"^dense_37/Tensordot/ReadVariableOp/^dense_37/kernel/Regularizer/Abs/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp ^dense_38/BiasAdd/ReadVariableOp^dense_38/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 2B
dense_36/BiasAdd/ReadVariableOpdense_36/BiasAdd/ReadVariableOp2F
!dense_36/Tensordot/ReadVariableOp!dense_36/Tensordot/ReadVariableOp2`
.dense_36/kernel/Regularizer/Abs/ReadVariableOp.dense_36/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2B
dense_37/BiasAdd/ReadVariableOpdense_37/BiasAdd/ReadVariableOp2F
!dense_37/Tensordot/ReadVariableOp!dense_37/Tensordot/ReadVariableOp2`
.dense_37/kernel/Regularizer/Abs/ReadVariableOp.dense_37/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2B
dense_38/BiasAdd/ReadVariableOpdense_38/BiasAdd/ReadVariableOp2@
dense_38/MatMul/ReadVariableOpdense_38/MatMul/ReadVariableOp:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
G
+__inference_dropout_6_layer_call_fn_1950227

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_19494402
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
d
F__inference_dropout_6_layer_call_and_return_conditional_losses_1950210

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:??????????@2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:??????????@2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
*__inference_dense_38_layer_call_fn_1950262

inputs
unknown:	?T
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_19494602
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????T: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????T
 
_user_specified_nameinputs
?	
?
/__inference_sequential_24_layer_call_fn_1949512
dense_36_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:	?T
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_36_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_24_layer_call_and_return_conditional_losses_19494972
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
,
_output_shapes
:??????????
(
_user_specified_namedense_36_input
?=
?
 __inference__traced_save_1950406
file_prefix.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_36_kernel_m_read_readvariableop3
/savev2_adam_dense_36_bias_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableop5
1savev2_adam_dense_38_kernel_m_read_readvariableop3
/savev2_adam_dense_38_bias_m_read_readvariableop5
1savev2_adam_dense_36_kernel_v_read_readvariableop3
/savev2_adam_dense_36_bias_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableop5
1savev2_adam_dense_38_kernel_v_read_readvariableop3
/savev2_adam_dense_38_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_36_kernel_m_read_readvariableop/savev2_adam_dense_36_bias_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableop1savev2_adam_dense_38_kernel_m_read_readvariableop/savev2_adam_dense_38_bias_m_read_readvariableop1savev2_adam_dense_36_kernel_v_read_readvariableop/savev2_adam_dense_36_bias_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableop1savev2_adam_dense_38_kernel_v_read_readvariableop/savev2_adam_dense_38_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:	?T:: : : : : : : : : :@:@:@@:@:	?T::@:@:@@:@:	?T:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:%!

_output_shapes
:	?T: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:%!

_output_shapes
:	?T: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:%!

_output_shapes
:	?T: 

_output_shapes
::

_output_shapes
: 
?

?
E__inference_dense_38_layer_call_and_return_conditional_losses_1950253

inputs1
matmul_readvariableop_resource:	?T-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?T*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????T: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????T
 
_user_specified_nameinputs
?	
?
/__inference_sequential_24_layer_call_fn_1950065

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:	?T
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_24_layer_call_and_return_conditional_losses_19496412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?H
?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949497

inputs"
dense_36_1949378:@
dense_36_1949380:@"
dense_37_1949430:@@
dense_37_1949432:@#
dense_38_1949461:	?T
dense_38_1949463:
identity?? dense_36/StatefulPartitionedCall?.dense_36/kernel/Regularizer/Abs/ReadVariableOp?1dense_36/kernel/Regularizer/Square/ReadVariableOp? dense_37/StatefulPartitionedCall?.dense_37/kernel/Regularizer/Abs/ReadVariableOp?1dense_37/kernel/Regularizer/Square/ReadVariableOp? dense_38/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_1949378dense_36_1949380*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_36_layer_call_and_return_conditional_losses_19493772"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_1949430dense_37_1949432*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_37_layer_call_and_return_conditional_losses_19494292"
 dense_37/StatefulPartitionedCall?
dropout_6/PartitionedCallPartitionedCall)dense_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_19494402
dropout_6/PartitionedCall?
flatten_24/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_24_layer_call_and_return_conditional_losses_19494482
flatten_24/PartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0dense_38_1949461dense_38_1949463*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_19494602"
 dense_38/StatefulPartitionedCall?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_1949378*
_output_shapes

:@*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_1949378*
_output_shapes

:@*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_1949430*
_output_shapes

:@@*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_1949430*
_output_shapes

:@@*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_36/StatefulPartitionedCall/^dense_36/kernel/Regularizer/Abs/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp!^dense_37/StatefulPartitionedCall/^dense_37/kernel/Regularizer/Abs/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2`
.dense_36/kernel/Regularizer/Abs/ReadVariableOp.dense_36/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2`
.dense_37/kernel/Regularizer/Abs/ReadVariableOp.dense_37/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?I
?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949724
dense_36_input"
dense_36_1949676:@
dense_36_1949678:@"
dense_37_1949681:@@
dense_37_1949683:@#
dense_38_1949688:	?T
dense_38_1949690:
identity?? dense_36/StatefulPartitionedCall?.dense_36/kernel/Regularizer/Abs/ReadVariableOp?1dense_36/kernel/Regularizer/Square/ReadVariableOp? dense_37/StatefulPartitionedCall?.dense_37/kernel/Regularizer/Abs/ReadVariableOp?1dense_37/kernel/Regularizer/Square/ReadVariableOp? dense_38/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_1949676dense_36_1949678*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_36_layer_call_and_return_conditional_losses_19493772"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_1949681dense_37_1949683*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_37_layer_call_and_return_conditional_losses_19494292"
 dense_37/StatefulPartitionedCall?
dropout_6/PartitionedCallPartitionedCall)dense_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_19494402
dropout_6/PartitionedCall?
flatten_24/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_24_layer_call_and_return_conditional_losses_19494482
flatten_24/PartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0dense_38_1949688dense_38_1949690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_19494602"
 dense_38/StatefulPartitionedCall?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_1949676*
_output_shapes

:@*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_1949676*
_output_shapes

:@*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_1949681*
_output_shapes

:@@*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_1949681*
_output_shapes

:@@*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_36/StatefulPartitionedCall/^dense_36/kernel/Regularizer/Abs/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp!^dense_37/StatefulPartitionedCall/^dense_37/kernel/Regularizer/Abs/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2`
.dense_36/kernel/Regularizer/Abs/ReadVariableOp.dense_36/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2`
.dense_37/kernel/Regularizer/Abs/ReadVariableOp.dense_37/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall:\ X
,
_output_shapes
:??????????
(
_user_specified_namedense_36_input
?J
?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949775
dense_36_input"
dense_36_1949727:@
dense_36_1949729:@"
dense_37_1949732:@@
dense_37_1949734:@#
dense_38_1949739:	?T
dense_38_1949741:
identity?? dense_36/StatefulPartitionedCall?.dense_36/kernel/Regularizer/Abs/ReadVariableOp?1dense_36/kernel/Regularizer/Square/ReadVariableOp? dense_37/StatefulPartitionedCall?.dense_37/kernel/Regularizer/Abs/ReadVariableOp?1dense_37/kernel/Regularizer/Square/ReadVariableOp? dense_38/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCalldense_36_inputdense_36_1949727dense_36_1949729*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_36_layer_call_and_return_conditional_losses_19493772"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_1949732dense_37_1949734*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_37_layer_call_and_return_conditional_losses_19494292"
 dense_37/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_19495482#
!dropout_6/StatefulPartitionedCall?
flatten_24/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_24_layer_call_and_return_conditional_losses_19494482
flatten_24/PartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0dense_38_1949739dense_38_1949741*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_19494602"
 dense_38/StatefulPartitionedCall?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_1949727*
_output_shapes

:@*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_1949727*
_output_shapes

:@*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_1949732*
_output_shapes

:@@*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_1949732*
_output_shapes

:@@*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_36/StatefulPartitionedCall/^dense_36/kernel/Regularizer/Abs/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp!^dense_37/StatefulPartitionedCall/^dense_37/kernel/Regularizer/Abs/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2`
.dense_36/kernel/Regularizer/Abs/ReadVariableOp.dense_36/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2`
.dense_37/kernel/Regularizer/Abs/ReadVariableOp.dense_37/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall:\ X
,
_output_shapes
:??????????
(
_user_specified_namedense_36_input
?
c
G__inference_flatten_24_layer_call_and_return_conditional_losses_1950238

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? *  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????T2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????T2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?J
?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949641

inputs"
dense_36_1949593:@
dense_36_1949595:@"
dense_37_1949598:@@
dense_37_1949600:@#
dense_38_1949605:	?T
dense_38_1949607:
identity?? dense_36/StatefulPartitionedCall?.dense_36/kernel/Regularizer/Abs/ReadVariableOp?1dense_36/kernel/Regularizer/Square/ReadVariableOp? dense_37/StatefulPartitionedCall?.dense_37/kernel/Regularizer/Abs/ReadVariableOp?1dense_37/kernel/Regularizer/Square/ReadVariableOp? dense_38/StatefulPartitionedCall?!dropout_6/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_1949593dense_36_1949595*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_36_layer_call_and_return_conditional_losses_19493772"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_1949598dense_37_1949600*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_37_layer_call_and_return_conditional_losses_19494292"
 dense_37/StatefulPartitionedCall?
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall)dense_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_19495482#
!dropout_6/StatefulPartitionedCall?
flatten_24/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_24_layer_call_and_return_conditional_losses_19494482
flatten_24/PartitionedCall?
 dense_38/StatefulPartitionedCallStatefulPartitionedCall#flatten_24/PartitionedCall:output:0dense_38_1949605dense_38_1949607*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_38_layer_call_and_return_conditional_losses_19494602"
 dense_38/StatefulPartitionedCall?
!dense_36/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_36/kernel/Regularizer/Const?
.dense_36/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_36_1949593*
_output_shapes

:@*
dtype020
.dense_36/kernel/Regularizer/Abs/ReadVariableOp?
dense_36/kernel/Regularizer/AbsAbs6dense_36/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@2!
dense_36/kernel/Regularizer/Abs?
#dense_36/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_1?
dense_36/kernel/Regularizer/SumSum#dense_36/kernel/Regularizer/Abs:y:0,dense_36/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/Sum?
!dense_36/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_36/kernel/Regularizer/mul/x?
dense_36/kernel/Regularizer/mulMul*dense_36/kernel/Regularizer/mul/x:output:0(dense_36/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/mul?
dense_36/kernel/Regularizer/addAddV2*dense_36/kernel/Regularizer/Const:output:0#dense_36/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_36/kernel/Regularizer/add?
1dense_36/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_36_1949593*
_output_shapes

:@*
dtype023
1dense_36/kernel/Regularizer/Square/ReadVariableOp?
"dense_36/kernel/Regularizer/SquareSquare9dense_36/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@2$
"dense_36/kernel/Regularizer/Square?
#dense_36/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_36/kernel/Regularizer/Const_2?
!dense_36/kernel/Regularizer/Sum_1Sum&dense_36/kernel/Regularizer/Square:y:0,dense_36/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/Sum_1?
#dense_36/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_36/kernel/Regularizer/mul_1/x?
!dense_36/kernel/Regularizer/mul_1Mul,dense_36/kernel/Regularizer/mul_1/x:output:0*dense_36/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/mul_1?
!dense_36/kernel/Regularizer/add_1AddV2#dense_36/kernel/Regularizer/add:z:0%dense_36/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_36/kernel/Regularizer/add_1?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_37_1949598*
_output_shapes

:@@*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_37_1949598*
_output_shapes

:@@*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1?
IdentityIdentity)dense_38/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp!^dense_36/StatefulPartitionedCall/^dense_36/kernel/Regularizer/Abs/ReadVariableOp2^dense_36/kernel/Regularizer/Square/ReadVariableOp!^dense_37/StatefulPartitionedCall/^dense_37/kernel/Regularizer/Abs/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp!^dense_38/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : 2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2`
.dense_36/kernel/Regularizer/Abs/ReadVariableOp.dense_36/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_36/kernel/Regularizer/Square/ReadVariableOp1dense_36/kernel/Regularizer/Square/ReadVariableOp2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall2`
.dense_37/kernel/Regularizer/Abs/ReadVariableOp.dense_37/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?u
?
#__inference__traced_restore_1950497
file_prefix2
 assignvariableop_dense_36_kernel:@.
 assignvariableop_1_dense_36_bias:@4
"assignvariableop_2_dense_37_kernel:@@.
 assignvariableop_3_dense_37_bias:@5
"assignvariableop_4_dense_38_kernel:	?T.
 assignvariableop_5_dense_38_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: <
*assignvariableop_15_adam_dense_36_kernel_m:@6
(assignvariableop_16_adam_dense_36_bias_m:@<
*assignvariableop_17_adam_dense_37_kernel_m:@@6
(assignvariableop_18_adam_dense_37_bias_m:@=
*assignvariableop_19_adam_dense_38_kernel_m:	?T6
(assignvariableop_20_adam_dense_38_bias_m:<
*assignvariableop_21_adam_dense_36_kernel_v:@6
(assignvariableop_22_adam_dense_36_bias_v:@<
*assignvariableop_23_adam_dense_37_kernel_v:@@6
(assignvariableop_24_adam_dense_37_bias_v:@=
*assignvariableop_25_adam_dense_38_kernel_v:	?T6
(assignvariableop_26_adam_dense_38_bias_v:
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_36_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_36_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_37_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_37_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_38_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_38_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_36_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_36_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_37_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_37_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_38_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_38_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_36_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_36_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_dense_37_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_dense_37_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_38_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_38_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27f
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_28?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?9
?
E__inference_dense_37_layer_call_and_return_conditional_losses_1949429

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?.dense_37/kernel/Regularizer/Abs/ReadVariableOp?1dense_37/kernel/Regularizer/Square/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:??????????@2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:??????????@2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:??????????@2	
BiasAdd]
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:??????????@2
Relu?
!dense_37/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_37/kernel/Regularizer/Const?
.dense_37/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype020
.dense_37/kernel/Regularizer/Abs/ReadVariableOp?
dense_37/kernel/Regularizer/AbsAbs6dense_37/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2!
dense_37/kernel/Regularizer/Abs?
#dense_37/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_1?
dense_37/kernel/Regularizer/SumSum#dense_37/kernel/Regularizer/Abs:y:0,dense_37/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/Sum?
!dense_37/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!dense_37/kernel/Regularizer/mul/x?
dense_37/kernel/Regularizer/mulMul*dense_37/kernel/Regularizer/mul/x:output:0(dense_37/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/mul?
dense_37/kernel/Regularizer/addAddV2*dense_37/kernel/Regularizer/Const:output:0#dense_37/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_37/kernel/Regularizer/add?
1dense_37/kernel/Regularizer/Square/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype023
1dense_37/kernel/Regularizer/Square/ReadVariableOp?
"dense_37/kernel/Regularizer/SquareSquare9dense_37/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:@@2$
"dense_37/kernel/Regularizer/Square?
#dense_37/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_37/kernel/Regularizer/Const_2?
!dense_37/kernel/Regularizer/Sum_1Sum&dense_37/kernel/Regularizer/Square:y:0,dense_37/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/Sum_1?
#dense_37/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#dense_37/kernel/Regularizer/mul_1/x?
!dense_37/kernel/Regularizer/mul_1Mul,dense_37/kernel/Regularizer/mul_1/x:output:0*dense_37/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/mul_1?
!dense_37/kernel/Regularizer/add_1AddV2#dense_37/kernel/Regularizer/add:z:0%dense_37/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_37/kernel/Regularizer/add_1r
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:??????????@2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp/^dense_37/kernel/Regularizer/Abs/ReadVariableOp2^dense_37/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp2`
.dense_37/kernel/Regularizer/Abs/ReadVariableOp.dense_37/kernel/Regularizer/Abs/ReadVariableOp2f
1dense_37/kernel/Regularizer/Square/ReadVariableOp1dense_37/kernel/Regularizer/Square/ReadVariableOp:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
d
+__inference_dropout_6_layer_call_fn_1950232

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_19495482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:??????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????@22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:??????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
N
dense_36_input<
 serving_default_dense_36_input:0??????????<
dense_380
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?p
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
*`&call_and_return_all_conditional_losses
a_default_save_signature
b__call__"
_tf_keras_sequential
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"
_tf_keras_layer
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"
_tf_keras_layer
?
	variables
regularization_losses
trainable_variables
	keras_api
*g&call_and_return_all_conditional_losses
h__call__"
_tf_keras_layer
?
	variables
regularization_losses
trainable_variables
	keras_api
*i&call_and_return_all_conditional_losses
j__call__"
_tf_keras_layer
?

 kernel
!bias
"	variables
#regularization_losses
$trainable_variables
%	keras_api
*k&call_and_return_all_conditional_losses
l__call__"
_tf_keras_layer
?
&iter

'beta_1

(beta_2
	)decay
*learning_ratemTmUmVmW mX!mYvZv[v\v] v^!v_"
	optimizer
J
0
1
2
3
 4
!5"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
?
+metrics
,layer_metrics
	variables

-layers
.layer_regularization_losses
/non_trainable_variables
regularization_losses
	trainable_variables
b__call__
a_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
,
oserving_default"
signature_map
!:@2dense_36/kernel
:@2dense_36/bias
.
0
1"
trackable_list_wrapper
'
m0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
0metrics
1layer_metrics
	variables

2layers
3layer_regularization_losses
4non_trainable_variables
regularization_losses
trainable_variables
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
!:@@2dense_37/kernel
:@2dense_37/bias
.
0
1"
trackable_list_wrapper
'
n0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
5metrics
6layer_metrics
	variables

7layers
8layer_regularization_losses
9non_trainable_variables
regularization_losses
trainable_variables
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
:metrics
;layer_metrics
	variables

<layers
=layer_regularization_losses
>non_trainable_variables
regularization_losses
trainable_variables
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
@layer_metrics
	variables

Alayers
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses
trainable_variables
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
": 	?T2dense_38/kernel
:2dense_38/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
Dmetrics
Elayer_metrics
"	variables

Flayers
Glayer_regularization_losses
Hnon_trainable_variables
#regularization_losses
$trainable_variables
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
I0
J1"
trackable_list_wrapper
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
m0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
n0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
N
	Ktotal
	Lcount
M	variables
N	keras_api"
_tf_keras_metric
^
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
K0
L1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
&:$@2Adam/dense_36/kernel/m
 :@2Adam/dense_36/bias/m
&:$@@2Adam/dense_37/kernel/m
 :@2Adam/dense_37/bias/m
':%	?T2Adam/dense_38/kernel/m
 :2Adam/dense_38/bias/m
&:$@2Adam/dense_36/kernel/v
 :@2Adam/dense_36/bias/v
&:$@@2Adam/dense_37/kernel/v
 :@2Adam/dense_37/bias/v
':%	?T2Adam/dense_38/kernel/v
 :2Adam/dense_38/bias/v
?2?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949927
J__inference_sequential_24_layer_call_and_return_conditional_losses_1950031
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949724
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949775?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_1949324dense_36_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_sequential_24_layer_call_fn_1949512
/__inference_sequential_24_layer_call_fn_1950048
/__inference_sequential_24_layer_call_fn_1950065
/__inference_sequential_24_layer_call_fn_1949673?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_dense_36_layer_call_and_return_conditional_losses_1950126?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_36_layer_call_fn_1950135?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_37_layer_call_and_return_conditional_losses_1950196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_37_layer_call_fn_1950205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout_6_layer_call_and_return_conditional_losses_1950210
F__inference_dropout_6_layer_call_and_return_conditional_losses_1950222?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_6_layer_call_fn_1950227
+__inference_dropout_6_layer_call_fn_1950232?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_flatten_24_layer_call_and_return_conditional_losses_1950238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_flatten_24_layer_call_fn_1950243?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_38_layer_call_and_return_conditional_losses_1950253?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_38_layer_call_fn_1950262?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_1950282?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_1950302?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
%__inference_signature_wrapper_1949830dense_36_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_1949324{ !<?9
2?/
-?*
dense_36_input??????????
? "3?0
.
dense_38"?
dense_38??????????
E__inference_dense_36_layer_call_and_return_conditional_losses_1950126f4?1
*?'
%?"
inputs??????????
? "*?'
 ?
0??????????@
? ?
*__inference_dense_36_layer_call_fn_1950135Y4?1
*?'
%?"
inputs??????????
? "???????????@?
E__inference_dense_37_layer_call_and_return_conditional_losses_1950196f4?1
*?'
%?"
inputs??????????@
? "*?'
 ?
0??????????@
? ?
*__inference_dense_37_layer_call_fn_1950205Y4?1
*?'
%?"
inputs??????????@
? "???????????@?
E__inference_dense_38_layer_call_and_return_conditional_losses_1950253] !0?-
&?#
!?
inputs??????????T
? "%?"
?
0?????????
? ~
*__inference_dense_38_layer_call_fn_1950262P !0?-
&?#
!?
inputs??????????T
? "???????????
F__inference_dropout_6_layer_call_and_return_conditional_losses_1950210f8?5
.?+
%?"
inputs??????????@
p 
? "*?'
 ?
0??????????@
? ?
F__inference_dropout_6_layer_call_and_return_conditional_losses_1950222f8?5
.?+
%?"
inputs??????????@
p
? "*?'
 ?
0??????????@
? ?
+__inference_dropout_6_layer_call_fn_1950227Y8?5
.?+
%?"
inputs??????????@
p 
? "???????????@?
+__inference_dropout_6_layer_call_fn_1950232Y8?5
.?+
%?"
inputs??????????@
p
? "???????????@?
G__inference_flatten_24_layer_call_and_return_conditional_losses_1950238^4?1
*?'
%?"
inputs??????????@
? "&?#
?
0??????????T
? ?
,__inference_flatten_24_layer_call_fn_1950243Q4?1
*?'
%?"
inputs??????????@
? "???????????T<
__inference_loss_fn_0_1950282?

? 
? "? <
__inference_loss_fn_1_1950302?

? 
? "? ?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949724u !D?A
:?7
-?*
dense_36_input??????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949775u !D?A
:?7
-?*
dense_36_input??????????
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1949927m !<?9
2?/
%?"
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_24_layer_call_and_return_conditional_losses_1950031m !<?9
2?/
%?"
inputs??????????
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_24_layer_call_fn_1949512h !D?A
:?7
-?*
dense_36_input??????????
p 

 
? "???????????
/__inference_sequential_24_layer_call_fn_1949673h !D?A
:?7
-?*
dense_36_input??????????
p

 
? "???????????
/__inference_sequential_24_layer_call_fn_1950048` !<?9
2?/
%?"
inputs??????????
p 

 
? "???????????
/__inference_sequential_24_layer_call_fn_1950065` !<?9
2?/
%?"
inputs??????????
p

 
? "???????????
%__inference_signature_wrapper_1949830? !N?K
? 
D?A
?
dense_36_input-?*
dense_36_input??????????"3?0
.
dense_38"?
dense_38?????????