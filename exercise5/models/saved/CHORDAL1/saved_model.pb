��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
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
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8��
t
gcn/VariableVarHandleOp*
shape
:d*
shared_namegcn/Variable*
dtype0*
_output_shapes
: 
m
 gcn/Variable/Read/ReadVariableOpReadVariableOpgcn/Variable*
dtype0*
_output_shapes

:d
t
dense/kernelVarHandleOp*
shape
:*
shared_namedense/kernel*
dtype0*
_output_shapes
: 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:
l

dense/biasVarHandleOp*
shape:*
shared_name
dense/bias*
dtype0*
_output_shapes
: 
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
f
	Adam/iterVarHandleOp*
shape: *
shared_name	Adam/iter*
dtype0	*
_output_shapes
: 
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
dtype0	*
_output_shapes
: 
j
Adam/beta_1VarHandleOp*
shape: *
shared_nameAdam/beta_1*
dtype0*
_output_shapes
: 
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
dtype0*
_output_shapes
: 
j
Adam/beta_2VarHandleOp*
shape: *
shared_nameAdam/beta_2*
dtype0*
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
dtype0*
_output_shapes
: 
h

Adam/decayVarHandleOp*
shape: *
shared_name
Adam/decay*
dtype0*
_output_shapes
: 
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
dtype0*
_output_shapes
: 
x
Adam/learning_rateVarHandleOp*
shape: *#
shared_nameAdam/learning_rate*
dtype0*
_output_shapes
: 
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shape: *
shared_nametotal*
dtype0*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
�
Adam/gcn/Variable/mVarHandleOp*
shape
:d*$
shared_nameAdam/gcn/Variable/m*
dtype0*
_output_shapes
: 
{
'Adam/gcn/Variable/m/Read/ReadVariableOpReadVariableOpAdam/gcn/Variable/m*
dtype0*
_output_shapes

:d
�
Adam/dense/kernel/mVarHandleOp*
shape
:*$
shared_nameAdam/dense/kernel/m*
dtype0*
_output_shapes
: 
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
dtype0*
_output_shapes

:
z
Adam/dense/bias/mVarHandleOp*
shape:*"
shared_nameAdam/dense/bias/m*
dtype0*
_output_shapes
: 
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
dtype0*
_output_shapes
:
�
Adam/gcn/Variable/vVarHandleOp*
shape
:d*$
shared_nameAdam/gcn/Variable/v*
dtype0*
_output_shapes
: 
{
'Adam/gcn/Variable/v/Read/ReadVariableOpReadVariableOpAdam/gcn/Variable/v*
dtype0*
_output_shapes

:d
�
Adam/dense/kernel/vVarHandleOp*
shape
:*$
shared_nameAdam/dense/kernel/v*
dtype0*
_output_shapes
: 
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
dtype0*
_output_shapes

:
z
Adam/dense/bias/vVarHandleOp*
shape:*"
shared_nameAdam/dense/bias/v*
dtype0*
_output_shapes
: 
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
�
ConstConst"/device:CPU:0*�
value�B� B�
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
Y
w
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
v
#iter

$beta_1

%beta_2
	&decay
'learning_ratemLmMmNvOvPvQ
 

0
1
2

0
1
2
�
(metrics
regularization_losses
trainable_variables

)layers
*layer_regularization_losses
+non_trainable_variables
		variables
 
 
 
 
�
,metrics
regularization_losses
trainable_variables

-layers
.layer_regularization_losses
/non_trainable_variables
	variables
 
 
 
�
0metrics
regularization_losses
trainable_variables

1layers
2layer_regularization_losses
3non_trainable_variables
	variables
SQ
VARIABLE_VALUEgcn/Variable1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
�
4metrics
regularization_losses
trainable_variables

5layers
6layer_regularization_losses
7non_trainable_variables
	variables
 
 
 
�
8metrics
regularization_losses
trainable_variables

9layers
:layer_regularization_losses
;non_trainable_variables
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
<metrics
regularization_losses
 trainable_variables

=layers
>layer_regularization_losses
?non_trainable_variables
!	variables
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

@0
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
x
	Atotal
	Bcount
C
_fn_kwargs
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

A0
B1
�
Hmetrics
Dregularization_losses
Etrainable_variables

Ilayers
Jlayer_regularization_losses
Knon_trainable_variables
F	variables
 
 
 

A0
B1
vt
VARIABLE_VALUEAdam/gcn/Variable/mMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/gcn/Variable/vMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
�
&serving_default_adjacency_matrix_inputPlaceholder* 
shape:���������dd*
dtype0*+
_output_shapes
:���������dd
�
"serving_default_node_feature_inputPlaceholder* 
shape:���������dd*
dtype0*+
_output_shapes
:���������dd
�
StatefulPartitionedCallStatefulPartitionedCall&serving_default_adjacency_matrix_input"serving_default_node_feature_inputgcn/Variabledense/kernel
dense/bias*,
_gradient_op_typePartitionedCall-11805*,
f'R%
#__inference_signature_wrapper_11583*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*'
_output_shapes
:���������
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename gcn/Variable/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/gcn/Variable/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp'Adam/gcn/Variable/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*,
_gradient_op_typePartitionedCall-11843*'
f"R 
__inference__traced_save_11842*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2	*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegcn/Variabledense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/gcn/Variable/mAdam/dense/kernel/mAdam/dense/bias/mAdam/gcn/Variable/vAdam/dense/kernel/vAdam/dense/bias/v*,
_gradient_op_typePartitionedCall-11904**
f%R#
!__inference__traced_restore_11903*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: ��
�
f
C__inference_sum_pool_layer_call_and_return_conditional_losses_11725
node_features
identityW
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: k
SumSumnode_featuresSum/reduction_indices:output:0*
T0*'
_output_shapes
:���������T
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������d:- )
'
_user_specified_namenode_features
�
�
>__inference_gcn_layer_call_and_return_conditional_losses_11377

inputs
inputs_1$
 matmul_1_readvariableop_resource
identity��MatMul_1/ReadVariableOp_
MatMulBatchMatMulV2inputs_1inputs*
T0*+
_output_shapes
:���������dd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������dU
ReluReluMatMul_1:output:0*
T0*+
_output_shapes
:���������dx
IdentityIdentityRelu:activations:0^MatMul_1/ReadVariableOp*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*E
_input_shapes4
2:���������dd:���������dd:22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:&"
 
_user_specified_nameinputs:& "
 
_user_specified_nameinputs: 
�
K
(__inference_sum_pool_layer_call_fn_11736
node_features
identity�
PartitionedCallPartitionedCallnode_features*,
_gradient_op_typePartitionedCall-11417*L
fGRE
C__inference_sum_pool_layer_call_and_return_conditional_losses_11405*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������d:- )
'
_user_specified_namenode_features
�
�
%__inference_model_layer_call_fn_11535
node_feature_input
adjacency_matrix_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnode_feature_inputadjacency_matrix_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-11528*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11527*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*)
_output_shapes
:���������: �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*M
_input_shapes<
::���������dd:���������dd:::22
StatefulPartitionedCallStatefulPartitionedCall: :62
0
_user_specified_nameadjacency_matrix_input:2 .
,
_user_specified_namenode_feature_input: : 
�
�
>__inference_gcn_layer_call_and_return_conditional_losses_11695
inputs_0
inputs_1$
 matmul_1_readvariableop_resource
identity��MatMul_1/ReadVariableOpa
MatMulBatchMatMulV2inputs_1inputs_0*
T0*+
_output_shapes
:���������dd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������dU
ReluReluMatMul_1:output:0*
T0*+
_output_shapes
:���������dx
IdentityIdentityRelu:activations:0^MatMul_1/ReadVariableOp*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*E
_input_shapes4
2:���������dd:���������dd:22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0: 
�

�
D__inference_dense_layer_call_and_return_all_conditional_losses_11768

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11447*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11441*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-11461*5
f0R.
,__inference_dense_activity_regularizer_11348*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������k

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
>__inference_gcn_layer_call_and_return_conditional_losses_11705
inputs_0
inputs_1$
 matmul_1_readvariableop_resource
identity��MatMul_1/ReadVariableOpa
MatMulBatchMatMulV2inputs_1inputs_0*
T0*+
_output_shapes
:���������dd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������dU
ReluReluMatMul_1:output:0*
T0*+
_output_shapes
:���������dx
IdentityIdentityRelu:activations:0^MatMul_1/ReadVariableOp*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*E
_input_shapes4
2:���������dd:���������dd:22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0: 
�
�
@__inference_model_layer_call_and_return_conditional_losses_11503
node_feature_input
adjacency_matrix_input&
"gcn_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity

identity_1��dense/StatefulPartitionedCall�gcn/StatefulPartitionedCall�
gcn/StatefulPartitionedCallStatefulPartitionedCallnode_feature_inputadjacency_matrix_input"gcn_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11392*G
fBR@
>__inference_gcn_layer_call_and_return_conditional_losses_11377*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:���������d�
sum_pool/PartitionedCallPartitionedCall$gcn/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-11425*L
fGRE
C__inference_sum_pool_layer_call_and_return_conditional_losses_11413*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
dense/StatefulPartitionedCallStatefulPartitionedCall!sum_pool/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11447*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11441*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-11461*5
f0R.
,__inference_dense_activity_regularizer_11348*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gcn/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall^gcn/StatefulPartitionedCall*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*M
_input_shapes<
::���������dd:���������dd:::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gcn/StatefulPartitionedCallgcn/StatefulPartitionedCall: :62
0
_user_specified_nameadjacency_matrix_input:2 .
,
_user_specified_namenode_feature_input: : 
�
K
(__inference_sum_pool_layer_call_fn_11741
node_features
identity�
PartitionedCallPartitionedCallnode_features*,
_gradient_op_typePartitionedCall-11425*L
fGRE
C__inference_sum_pool_layer_call_and_return_conditional_losses_11413*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:���������`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������d:- )
'
_user_specified_namenode_features
�
�
%__inference_dense_layer_call_fn_11759

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11447*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11441*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
f
C__inference_sum_pool_layer_call_and_return_conditional_losses_11405
node_features
identityW
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: k
SumSumnode_featuresSum/reduction_indices:output:0*
T0*'
_output_shapes
:���������T
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������d:- )
'
_user_specified_namenode_features
�*
�
@__inference_model_layer_call_and_return_conditional_losses_11665
inputs_0
inputs_1(
$gcn_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity

identity_1��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�gcn/MatMul_1/ReadVariableOpe

gcn/MatMulBatchMatMulV2inputs_1inputs_0*
T0*+
_output_shapes
:���������dd�
gcn/MatMul_1/ReadVariableOpReadVariableOp$gcn_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d�
gcn/MatMul_1BatchMatMulV2gcn/MatMul:output:0#gcn/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d]
gcn/ReluRelugcn/MatMul_1:output:0*
T0*+
_output_shapes
:���������d`
sum_pool/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: �
sum_pool/SumSumgcn/Relu:activations:0'sum_pool/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:�
dense/MatMulMatMulsum_pool/Sum:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������o
dense/ActivityRegularizer/AbsAbsdense/Softmax:softmax:0*
T0*'
_output_shapes
:���������p
dense/ActivityRegularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:�
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
valueB
 *���=*
dtype0*
_output_shapes
: �
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
dense/ActivityRegularizer/addAddV2(dense/ActivityRegularizer/add/x:output:0!dense/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: u
 dense/ActivityRegularizer/SquareSquaredense/Softmax:softmax:0*
T0*'
_output_shapes
:���������r
!dense/ActivityRegularizer/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:�
dense/ActivityRegularizer/Sum_1Sum$dense/ActivityRegularizer/Square:y:0*dense/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense/ActivityRegularizer/mul_1/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
dense/ActivityRegularizer/mul_1Mul*dense/ActivityRegularizer/mul_1/x:output:0(dense/ActivityRegularizer/Sum_1:output:0*
T0*
_output_shapes
: �
dense/ActivityRegularizer/add_1AddV2!dense/ActivityRegularizer/add:z:0#dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: f
dense/ActivityRegularizer/ShapeShapedense/Softmax:softmax:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv#dense/ActivityRegularizer/add_1:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
IdentityIdentitydense/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^gcn/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^gcn/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*M
_input_shapes<
::���������dd:���������dd:::2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
gcn/MatMul_1/ReadVariableOpgcn/MatMul_1/ReadVariableOp: :($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0: : 
�,
�
 __inference__wrapped_model_11322
node_feature_input
adjacency_matrix_input.
*model_gcn_matmul_1_readvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource
identity��"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�!model/gcn/MatMul_1/ReadVariableOp�
model/gcn/MatMulBatchMatMulV2adjacency_matrix_inputnode_feature_input*
T0*+
_output_shapes
:���������dd�
!model/gcn/MatMul_1/ReadVariableOpReadVariableOp*model_gcn_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d�
model/gcn/MatMul_1BatchMatMulV2model/gcn/MatMul:output:0)model/gcn/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������di
model/gcn/ReluRelumodel/gcn/MatMul_1:output:0*
T0*+
_output_shapes
:���������df
$model/sum_pool/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: �
model/sum_pool/SumSummodel/gcn/Relu:activations:0-model/sum_pool/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:�
model/dense/MatMulMatMulmodel/sum_pool/Sum:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
model/dense/SoftmaxSoftmaxmodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������{
#model/dense/ActivityRegularizer/AbsAbsmodel/dense/Softmax:softmax:0*
T0*'
_output_shapes
:���������v
%model/dense/ActivityRegularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:�
#model/dense/ActivityRegularizer/SumSum'model/dense/ActivityRegularizer/Abs:y:0.model/dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: j
%model/dense/ActivityRegularizer/mul/xConst*
valueB
 *���=*
dtype0*
_output_shapes
: �
#model/dense/ActivityRegularizer/mulMul.model/dense/ActivityRegularizer/mul/x:output:0,model/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: j
%model/dense/ActivityRegularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
#model/dense/ActivityRegularizer/addAddV2.model/dense/ActivityRegularizer/add/x:output:0'model/dense/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: �
&model/dense/ActivityRegularizer/SquareSquaremodel/dense/Softmax:softmax:0*
T0*'
_output_shapes
:���������x
'model/dense/ActivityRegularizer/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:�
%model/dense/ActivityRegularizer/Sum_1Sum*model/dense/ActivityRegularizer/Square:y:00model/dense/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: l
'model/dense/ActivityRegularizer/mul_1/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
%model/dense/ActivityRegularizer/mul_1Mul0model/dense/ActivityRegularizer/mul_1/x:output:0.model/dense/ActivityRegularizer/Sum_1:output:0*
T0*
_output_shapes
: �
%model/dense/ActivityRegularizer/add_1AddV2'model/dense/ActivityRegularizer/add:z:0)model/dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: r
%model/dense/ActivityRegularizer/ShapeShapemodel/dense/Softmax:softmax:0*
T0*
_output_shapes
:}
3model/dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
5model/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
5model/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
-model/dense/ActivityRegularizer/strided_sliceStridedSlice.model/dense/ActivityRegularizer/Shape:output:0<model/dense/ActivityRegularizer/strided_slice/stack:output:0>model/dense/ActivityRegularizer/strided_slice/stack_1:output:0>model/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
$model/dense/ActivityRegularizer/CastCast6model/dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: �
'model/dense/ActivityRegularizer/truedivRealDiv)model/dense/ActivityRegularizer/add_1:z:0(model/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
IdentityIdentitymodel/dense/Softmax:softmax:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp"^model/gcn/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*M
_input_shapes<
::���������dd:���������dd:::2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/gcn/MatMul_1/ReadVariableOp!model/gcn/MatMul_1/ReadVariableOp: :62
0
_user_specified_nameadjacency_matrix_input:2 .
,
_user_specified_namenode_feature_input: : 
�
�
%__inference_model_layer_call_fn_11675
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-11528*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11527*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*)
_output_shapes
:���������: �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*M
_input_shapes<
::���������dd:���������dd:::22
StatefulPartitionedCallStatefulPartitionedCall: :($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0: : 
�
�
%__inference_model_layer_call_fn_11685
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-11561*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11560*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*)
_output_shapes
:���������: �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*M
_input_shapes<
::���������dd:���������dd:::22
StatefulPartitionedCallStatefulPartitionedCall: :($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0: : 
�
�
#__inference_signature_wrapper_11583
adjacency_matrix_input
node_feature_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnode_feature_inputadjacency_matrix_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-11577*)
f$R"
 __inference__wrapped_model_11322*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*M
_input_shapes<
::���������dd:���������dd:::22
StatefulPartitionedCallStatefulPartitionedCall: :2.
,
_user_specified_namenode_feature_input:6 2
0
_user_specified_nameadjacency_matrix_input: : 
�	
�
@__inference_dense_layer_call_and_return_conditional_losses_11441

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�	
�
@__inference_dense_layer_call_and_return_conditional_losses_11752

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�*
�
@__inference_model_layer_call_and_return_conditional_losses_11625
inputs_0
inputs_1(
$gcn_matmul_1_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity

identity_1��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�gcn/MatMul_1/ReadVariableOpe

gcn/MatMulBatchMatMulV2inputs_1inputs_0*
T0*+
_output_shapes
:���������dd�
gcn/MatMul_1/ReadVariableOpReadVariableOp$gcn_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d�
gcn/MatMul_1BatchMatMulV2gcn/MatMul:output:0#gcn/MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������d]
gcn/ReluRelugcn/MatMul_1:output:0*
T0*+
_output_shapes
:���������d`
sum_pool/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: �
sum_pool/SumSumgcn/Relu:activations:0'sum_pool/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:����������
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:�
dense/MatMulMatMulsum_pool/Sum:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������b
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������o
dense/ActivityRegularizer/AbsAbsdense/Softmax:softmax:0*
T0*'
_output_shapes
:���������p
dense/ActivityRegularizer/ConstConst*
valueB"       *
dtype0*
_output_shapes
:�
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0(dense/ActivityRegularizer/Const:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/mul/xConst*
valueB
 *���=*
dtype0*
_output_shapes
: �
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: d
dense/ActivityRegularizer/add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: �
dense/ActivityRegularizer/addAddV2(dense/ActivityRegularizer/add/x:output:0!dense/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: u
 dense/ActivityRegularizer/SquareSquaredense/Softmax:softmax:0*
T0*'
_output_shapes
:���������r
!dense/ActivityRegularizer/Const_1Const*
valueB"       *
dtype0*
_output_shapes
:�
dense/ActivityRegularizer/Sum_1Sum$dense/ActivityRegularizer/Square:y:0*dense/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: f
!dense/ActivityRegularizer/mul_1/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: �
dense/ActivityRegularizer/mul_1Mul*dense/ActivityRegularizer/mul_1/x:output:0(dense/ActivityRegularizer/Sum_1:output:0*
T0*
_output_shapes
: �
dense/ActivityRegularizer/add_1AddV2!dense/ActivityRegularizer/add:z:0#dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: f
dense/ActivityRegularizer/ShapeShapedense/Softmax:softmax:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv#dense/ActivityRegularizer/add_1:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
IdentityIdentitydense/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^gcn/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^gcn/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*M
_input_shapes<
::���������dd:���������dd:::2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
gcn/MatMul_1/ReadVariableOpgcn/MatMul_1/ReadVariableOp: :($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0: : 
�
f
C__inference_sum_pool_layer_call_and_return_conditional_losses_11731
node_features
identityW
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: k
SumSumnode_featuresSum/reduction_indices:output:0*
T0*'
_output_shapes
:���������T
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������d:- )
'
_user_specified_namenode_features
�
f
C__inference_sum_pool_layer_call_and_return_conditional_losses_11413
node_features
identityW
Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: k
SumSumnode_featuresSum/reduction_indices:output:0*
T0*'
_output_shapes
:���������T
IdentityIdentitySum:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0**
_input_shapes
:���������d:- )
'
_user_specified_namenode_features
�
F
,__inference_dense_activity_regularizer_11348
self
identity3
AbsAbsself*
T0*
_output_shapes
:6
RankRankAbs:y:0*
T0*
_output_shapes
: M
range/startConst*
value	B : *
dtype0*
_output_shapes
: M
range/deltaConst*
value	B :*
dtype0*
_output_shapes
: n
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:���������D
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: J
mul/xConst*
valueB
 *���=*
dtype0*
_output_shapes
: I
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: J
add/xConst*
valueB
 *    *
dtype0*
_output_shapes
: F
addAddV2add/x:output:0mul:z:0*
T0*
_output_shapes
: 9
SquareSquareself*
T0*
_output_shapes
:;
Rank_1Rank
Square:y:0*
T0*
_output_shapes
: O
range_1/startConst*
value	B : *
dtype0*
_output_shapes
: O
range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: v
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*#
_output_shapes
:���������K
Sum_1Sum
Square:y:0range_1:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
valueB
 *
�#<*
dtype0*
_output_shapes
: O
mul_1Mulmul_1/x:output:0Sum_1:output:0*
T0*
_output_shapes
: C
add_1AddV2add:z:0	mul_1:z:0*
T0*
_output_shapes
: @
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: "
identityIdentity:output:0*
_input_shapes
::$  

_user_specified_nameself
�
�
@__inference_model_layer_call_and_return_conditional_losses_11527

inputs
inputs_1&
"gcn_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity

identity_1��dense/StatefulPartitionedCall�gcn/StatefulPartitionedCall�
gcn/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1"gcn_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11382*G
fBR@
>__inference_gcn_layer_call_and_return_conditional_losses_11365*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:���������d�
sum_pool/PartitionedCallPartitionedCall$gcn/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-11417*L
fGRE
C__inference_sum_pool_layer_call_and_return_conditional_losses_11405*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
dense/StatefulPartitionedCallStatefulPartitionedCall!sum_pool/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11447*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11441*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-11461*5
f0R.
,__inference_dense_activity_regularizer_11348*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gcn/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall^gcn/StatefulPartitionedCall*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*M
_input_shapes<
::���������dd:���������dd:::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gcn/StatefulPartitionedCallgcn/StatefulPartitionedCall: :&"
 
_user_specified_nameinputs:& "
 
_user_specified_nameinputs: : 
�@
�
!__inference__traced_restore_11903
file_prefix!
assignvariableop_gcn_variable#
assignvariableop_1_dense_kernel!
assignvariableop_2_dense_bias 
assignvariableop_3_adam_iter"
assignvariableop_4_adam_beta_1"
assignvariableop_5_adam_beta_2!
assignvariableop_6_adam_decay)
%assignvariableop_7_adam_learning_rate
assignvariableop_8_total
assignvariableop_9_count+
'assignvariableop_10_adam_gcn_variable_m+
'assignvariableop_11_adam_dense_kernel_m)
%assignvariableop_12_adam_dense_bias_m+
'assignvariableop_13_adam_gcn_variable_v+
'assignvariableop_14_adam_dense_kernel_v)
%assignvariableop_15_adam_dense_bias_v
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
RestoreV2/shape_and_slicesConst"/device:CPU:0*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2	*T
_output_shapesB
@::::::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:y
AssignVariableOpAssignVariableOpassignvariableop_gcn_variableIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_kernelIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:}
AssignVariableOp_2AssignVariableOpassignvariableop_2_dense_biasIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0	*
_output_shapes
:|
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0*
dtype0	*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:~
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:~
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:}
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:x
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:x
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_gcn_variable_mIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp'assignvariableop_11_adam_dense_kernel_mIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp%assignvariableop_12_adam_dense_bias_mIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_adam_gcn_variable_vIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_kernel_vIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp%assignvariableop_15_adam_dense_bias_vIdentity_15:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: �
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6: : : : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : : :
 
�
�
#__inference_gcn_layer_call_fn_11712
inputs_0
inputs_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11382*G
fBR@
>__inference_gcn_layer_call_and_return_conditional_losses_11365*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:���������d�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*E
_input_shapes4
2:���������dd:���������dd:22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0: 
�
�
>__inference_gcn_layer_call_and_return_conditional_losses_11365

inputs
inputs_1$
 matmul_1_readvariableop_resource
identity��MatMul_1/ReadVariableOp_
MatMulBatchMatMulV2inputs_1inputs*
T0*+
_output_shapes
:���������dd�
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:d�
MatMul_1BatchMatMulV2MatMul:output:0MatMul_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������dU
ReluReluMatMul_1:output:0*
T0*+
_output_shapes
:���������dx
IdentityIdentityRelu:activations:0^MatMul_1/ReadVariableOp*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*E
_input_shapes4
2:���������dd:���������dd:22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:&"
 
_user_specified_nameinputs:& "
 
_user_specified_nameinputs: 
�
�
@__inference_model_layer_call_and_return_conditional_losses_11560

inputs
inputs_1&
"gcn_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity

identity_1��dense/StatefulPartitionedCall�gcn/StatefulPartitionedCall�
gcn/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1"gcn_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11392*G
fBR@
>__inference_gcn_layer_call_and_return_conditional_losses_11377*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:���������d�
sum_pool/PartitionedCallPartitionedCall$gcn/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-11425*L
fGRE
C__inference_sum_pool_layer_call_and_return_conditional_losses_11413*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
dense/StatefulPartitionedCallStatefulPartitionedCall!sum_pool/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11447*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11441*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-11461*5
f0R.
,__inference_dense_activity_regularizer_11348*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gcn/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall^gcn/StatefulPartitionedCall*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*M
_input_shapes<
::���������dd:���������dd:::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gcn/StatefulPartitionedCallgcn/StatefulPartitionedCall: :&"
 
_user_specified_nameinputs:& "
 
_user_specified_nameinputs: : 
�
�
@__inference_model_layer_call_and_return_conditional_losses_11481
node_feature_input
adjacency_matrix_input&
"gcn_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity

identity_1��dense/StatefulPartitionedCall�gcn/StatefulPartitionedCall�
gcn/StatefulPartitionedCallStatefulPartitionedCallnode_feature_inputadjacency_matrix_input"gcn_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11382*G
fBR@
>__inference_gcn_layer_call_and_return_conditional_losses_11365*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:���������d�
sum_pool/PartitionedCallPartitionedCall$gcn/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-11417*L
fGRE
C__inference_sum_pool_layer_call_and_return_conditional_losses_11405*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
dense/StatefulPartitionedCallStatefulPartitionedCall!sum_pool/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11447*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11441*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:����������
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-11461*5
f0R.
,__inference_dense_activity_regularizer_11348*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*
_output_shapes
: u
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:w
-dense/ActivityRegularizer/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:y
/dense/ActivityRegularizer/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:�
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: �
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

SrcT0*

DstT0*
_output_shapes
: �
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: �
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^gcn/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall^gcn/StatefulPartitionedCall*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0*M
_input_shapes<
::���������dd:���������dd:::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2:
gcn/StatefulPartitionedCallgcn/StatefulPartitionedCall: :62
0
_user_specified_nameadjacency_matrix_input:2 .
,
_user_specified_namenode_feature_input: : 
�
�
#__inference_gcn_layer_call_fn_11719
inputs_0
inputs_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11392*G
fBR@
>__inference_gcn_layer_call_and_return_conditional_losses_11377*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin
2*+
_output_shapes
:���������d�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:���������d"
identityIdentity:output:0*E
_input_shapes4
2:���������dd:���������dd:22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
inputs/1:( $
"
_user_specified_name
inputs/0: 
�(
�
__inference__traced_save_11842
file_prefix+
'savev2_gcn_variable_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_gcn_variable_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop2
.savev2_adam_gcn_variable_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_70ad33e4998941d78d47454031ed768c/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�B1layer_with_weights-0/w/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMlayer_with_weights-0/w/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
SaveV2/shape_and_slicesConst"/device:CPU:0*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_gcn_variable_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_gcn_variable_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop.savev2_adam_gcn_variable_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop"/device:CPU:0*
dtypes
2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*s
_input_shapesb
`: :d::: : : : : : : :d:::d::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : : : : :	 : : : : :+ '
%
_user_specified_namefile_prefix: : : : :
 
�
�
%__inference_model_layer_call_fn_11568
node_feature_input
adjacency_matrix_input"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnode_feature_inputadjacency_matrix_inputstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-11561*I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_11560*
Tout
2**
config_proto

CPU

GPU 2J 8*
Tin	
2*)
_output_shapes
:���������: �
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*M
_input_shapes<
::���������dd:���������dd:::22
StatefulPartitionedCallStatefulPartitionedCall: :62
0
_user_specified_nameadjacency_matrix_input:2 .
,
_user_specified_namenode_feature_input: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
U
node_feature_input?
$serving_default_node_feature_input:0���������dd
]
adjacency_matrix_inputC
(serving_default_adjacency_matrix_input:0���������dd9
dense0
StatefulPartitionedCall:0���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
� 
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
R_default_save_signature
*S&call_and_return_all_conditional_losses
T__call__"�
_tf_keras_model�{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 100, 100], "dtype": "float32", "sparse": false, "name": "node_feature_input"}, "name": "node_feature_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 100, 100], "dtype": "float32", "sparse": false, "name": "adjacency_matrix_input"}, "name": "adjacency_matrix_input", "inbound_nodes": []}, {"class_name": "GCN", "config": {"name": "gcn", "trainable": true, "dtype": "float32", "feature_num": 16}, "name": "gcn", "inbound_nodes": [[["node_feature_input", 0, 0, {}], ["adjacency_matrix_input", 0, 0, {}]]]}, {"class_name": "SumPool", "config": {"name": "sum_pool", "trainable": true, "dtype": "float32", "num_outputs": 16}, "name": "sum_pool", "inbound_nodes": [[["gcn", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 0.10000000149011612, "l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["sum_pool", 0, 0, {}]]]}], "input_layers": [["node_feature_input", 0, 0], ["adjacency_matrix_input", 0, 0]], "output_layers": [["dense", 0, 0]]}, "input_spec": [null, null], "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 100, 100], "dtype": "float32", "sparse": false, "name": "node_feature_input"}, "name": "node_feature_input", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 100, 100], "dtype": "float32", "sparse": false, "name": "adjacency_matrix_input"}, "name": "adjacency_matrix_input", "inbound_nodes": []}, {"class_name": "GCN", "config": {"name": "gcn", "trainable": true, "dtype": "float32", "feature_num": 16}, "name": "gcn", "inbound_nodes": [[["node_feature_input", 0, 0, {}], ["adjacency_matrix_input", 0, 0, {}]]]}, {"class_name": "SumPool", "config": {"name": "sum_pool", "trainable": true, "dtype": "float32", "num_outputs": 16}, "name": "sum_pool", "inbound_nodes": [[["gcn", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 0.10000000149011612, "l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["sum_pool", 0, 0, {}]]]}], "input_layers": [["node_feature_input", 0, 0], ["adjacency_matrix_input", 0, 0]], "output_layers": [["dense", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
regularization_losses
trainable_variables
	variables
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "node_feature_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 100, 100], "config": {"batch_input_shape": [null, 100, 100], "dtype": "float32", "sparse": false, "name": "node_feature_input"}}
�
regularization_losses
trainable_variables
	variables
	keras_api
*W&call_and_return_all_conditional_losses
X__call__"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "adjacency_matrix_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 100, 100], "config": {"batch_input_shape": [null, 100, 100], "dtype": "float32", "sparse": false, "name": "adjacency_matrix_input"}}
�
w
regularization_losses
trainable_variables
	variables
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"�
_tf_keras_layer�{"class_name": "GCN", "name": "gcn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "gcn", "trainable": true, "dtype": "float32", "feature_num": 16}}
�
regularization_losses
trainable_variables
	variables
	keras_api
*[&call_and_return_all_conditional_losses
\__call__"�
_tf_keras_layer�{"class_name": "SumPool", "name": "sum_pool", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sum_pool", "trainable": true, "dtype": "float32", "num_outputs": 16}}
�

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
*]&call_and_return_all_conditional_losses
^__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 0.10000000149011612, "l2": 0.009999999776482582}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 0.10000000149011612, "l2": 0.009999999776482582}}}
�
#iter

$beta_1

%beta_2
	&decay
'learning_ratemLmMmNvOvPvQ"
	optimizer
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
�
(metrics
regularization_losses
trainable_variables

)layers
*layer_regularization_losses
+non_trainable_variables
		variables
T__call__
R_default_save_signature
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
,
_serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
,metrics
regularization_losses
trainable_variables

-layers
.layer_regularization_losses
/non_trainable_variables
	variables
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
0metrics
regularization_losses
trainable_variables

1layers
2layer_regularization_losses
3non_trainable_variables
	variables
X__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
:d2gcn/Variable
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
4metrics
regularization_losses
trainable_variables

5layers
6layer_regularization_losses
7non_trainable_variables
	variables
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
8metrics
regularization_losses
trainable_variables

9layers
:layer_regularization_losses
;non_trainable_variables
	variables
\__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
<metrics
regularization_losses
 trainable_variables

=layers
>layer_regularization_losses
?non_trainable_variables
!	variables
^__call__
`activity_regularizer_fn
*]&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
'
@0"
trackable_list_wrapper
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	Atotal
	Bcount
C
_fn_kwargs
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
*b&call_and_return_all_conditional_losses
c__call__"�
_tf_keras_layer�{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
�
Hmetrics
Dregularization_losses
Etrainable_variables

Ilayers
Jlayer_regularization_losses
Knon_trainable_variables
F	variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
#:!d2Adam/gcn/Variable/m
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
#:!d2Adam/gcn/Variable/v
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
�2�
 __inference__wrapped_model_11322�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *p�m
k�h
0�-
node_feature_input���������dd
4�1
adjacency_matrix_input���������dd
�2�
@__inference_model_layer_call_and_return_conditional_losses_11503
@__inference_model_layer_call_and_return_conditional_losses_11625
@__inference_model_layer_call_and_return_conditional_losses_11665
@__inference_model_layer_call_and_return_conditional_losses_11481�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_model_layer_call_fn_11535
%__inference_model_layer_call_fn_11685
%__inference_model_layer_call_fn_11675
%__inference_model_layer_call_fn_11568�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
>__inference_gcn_layer_call_and_return_conditional_losses_11695
>__inference_gcn_layer_call_and_return_conditional_losses_11705�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
#__inference_gcn_layer_call_fn_11719
#__inference_gcn_layer_call_fn_11712�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
C__inference_sum_pool_layer_call_and_return_conditional_losses_11725
C__inference_sum_pool_layer_call_and_return_conditional_losses_11731�
���
FullArgSpec$
args�
jself
jnode_features
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
(__inference_sum_pool_layer_call_fn_11736
(__inference_sum_pool_layer_call_fn_11741�
���
FullArgSpec$
args�
jself
jnode_features
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
D__inference_dense_layer_call_and_return_all_conditional_losses_11768�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_dense_layer_call_fn_11759�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
SBQ
#__inference_signature_wrapper_11583adjacency_matrix_inputnode_feature_input
�2�
,__inference_dense_activity_regularizer_11348�
���
FullArgSpec
args�
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *�
	�
�2�
@__inference_dense_layer_call_and_return_conditional_losses_11752�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 �
>__inference_gcn_layer_call_and_return_conditional_losses_11695�r�o
X�U
S�P
&�#
inputs/0���������dd
&�#
inputs/1���������dd
�

trainingp")�&
�
0���������d
� �
D__inference_dense_layer_call_and_return_all_conditional_losses_11768j/�,
%�"
 �
inputs���������
� "3�0
�
0���������
�
�	
1/0 �
@__inference_model_layer_call_and_return_conditional_losses_11503���
x�u
k�h
0�-
node_feature_input���������dd
4�1
adjacency_matrix_input���������dd
p 

 
� "3�0
�
0���������
�
�	
1/0 �
(__inference_sum_pool_layer_call_fn_11736fJ�G
0�-
+�(
node_features���������d
�

trainingp"�����������
(__inference_sum_pool_layer_call_fn_11741fJ�G
0�-
+�(
node_features���������d
�

trainingp "����������x
%__inference_dense_layer_call_fn_11759O/�,
%�"
 �
inputs���������
� "�����������
C__inference_sum_pool_layer_call_and_return_conditional_losses_11725sJ�G
0�-
+�(
node_features���������d
�

trainingp"%�"
�
0���������
� �
%__inference_model_layer_call_fn_11535���
x�u
k�h
0�-
node_feature_input���������dd
4�1
adjacency_matrix_input���������dd
p

 
� "�����������
C__inference_sum_pool_layer_call_and_return_conditional_losses_11731sJ�G
0�-
+�(
node_features���������d
�

trainingp "%�"
�
0���������
� �
@__inference_model_layer_call_and_return_conditional_losses_11625�j�g
`�]
S�P
&�#
inputs/0���������dd
&�#
inputs/1���������dd
p

 
� "3�0
�
0���������
�
�	
1/0 Y
,__inference_dense_activity_regularizer_11348)�
�
�
self
� "� �
#__inference_signature_wrapper_11583����
� 
���
F
node_feature_input0�-
node_feature_input���������dd
N
adjacency_matrix_input4�1
adjacency_matrix_input���������dd"-�*
(
dense�
dense����������
>__inference_gcn_layer_call_and_return_conditional_losses_11705�r�o
X�U
S�P
&�#
inputs/0���������dd
&�#
inputs/1���������dd
�

trainingp ")�&
�
0���������d
� �
#__inference_gcn_layer_call_fn_11712�r�o
X�U
S�P
&�#
inputs/0���������dd
&�#
inputs/1���������dd
�

trainingp"����������d�
@__inference_model_layer_call_and_return_conditional_losses_11481���
x�u
k�h
0�-
node_feature_input���������dd
4�1
adjacency_matrix_input���������dd
p

 
� "3�0
�
0���������
�
�	
1/0 �
@__inference_dense_layer_call_and_return_conditional_losses_11752\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
 __inference__wrapped_model_11322�z�w
p�m
k�h
0�-
node_feature_input���������dd
4�1
adjacency_matrix_input���������dd
� "-�*
(
dense�
dense����������
%__inference_model_layer_call_fn_11675�j�g
`�]
S�P
&�#
inputs/0���������dd
&�#
inputs/1���������dd
p

 
� "�����������
#__inference_gcn_layer_call_fn_11719�r�o
X�U
S�P
&�#
inputs/0���������dd
&�#
inputs/1���������dd
�

trainingp "����������d�
%__inference_model_layer_call_fn_11685�j�g
`�]
S�P
&�#
inputs/0���������dd
&�#
inputs/1���������dd
p 

 
� "�����������
%__inference_model_layer_call_fn_11568���
x�u
k�h
0�-
node_feature_input���������dd
4�1
adjacency_matrix_input���������dd
p 

 
� "�����������
@__inference_model_layer_call_and_return_conditional_losses_11665�j�g
`�]
S�P
&�#
inputs/0���������dd
&�#
inputs/1���������dd
p 

 
� "3�0
�
0���������
�
�	
1/0 