Skip to left side bar




Filter files by name
/
Name
Last Modified










Code


Python 3
<a href="https://www.nvidia.com/dli"><img src="images/DLI_Header.png" alt="Header" style="width: 400px;"/></a>
## Assessment: Building a Real-Time Video AI Application ##
In this notebook, you will utilize what you've learned in this course to complete an assessment. The assessment has been divided into a couple of steps - each of which will generate a text file for grading purposes. You will be graded based on the following rubric. Note that this coding portion does not give partial credit - it shows up as either 0 or 60 points. Earning 50 points or more will award you the full 60 points, while earning less than 50 points will award you 0 points for the coding portion. 
<table border="1" class="dataframe" align='left'>  <thead>    <tr style="text-align: right;">      <th>Step</th>      <th># of &lt;FIXME&gt;</th>      <th>Points</th>    </tr>  </thead>  <tbody>    <tr>      <td>0. The Problem</td>      <td>0</td>      <td>0</td>    </tr>    <tr>      <td>1. Understanding the Input Video</td>      <td>5</td>      <td>10</td>    </tr>    <tr>      <td>2. Brainstorm AI Inference and Download a Pre-Trained Model</td>      <td>2</td>      <td>10</td>    </tr>    <tr>      <td>3. Edit the Inference Configuration File</td>      <td>10</td>      <td>10</td>    </tr>    <tr>      <td>4. Build and Run DeepStream Pipeline</td>      <td>20</td>      <td>20</td>    </tr>    <tr>      <td>5. Analyze the Results</td>      <td>1</td>      <td>10</td>    </tr>    <tr>      <td>BONUS. Visualize Frames</td>      <td>0</td>      <td>0</td>    </tr>  </tbody></table>
​
<p><img src='images/iva_framework.png' width=600></p>
### Step 0: The Problem ###
You are a developer for an automobile fleet management company. You have recently installed dashboard cameras on all of the vehicles and are ready to implement AI to analyze the fleet's driving behavior. One of the issues you've noticed with the fleet is [tailgating](https://en.wikipedia.org/wiki/Tailgating), which occurs when the vehicle drives behind another vehicle without leaving sufficient distance to stop without causing a collision if the vehicle in front stops suddenly. You've decied to build a DeepStream application that will help monitor this behavior. At this point, you want to be able to log occurences of tailgating so you can understand the frequency. Note that while the input video sources are static files, the pipeline can easily be modified to consume videos in real-time. 
**Instructions**: <br>
0.1 Execute the cell to set the target video as an environment variable. <br>
# 0.1
# DO NOT CHANGE THIS CELL
import os
os.environ['TARGET_VIDEO_PATH']='data/assessment_stream.h264'
os.environ['TARGET_VIDEO_PATH_MP4']='data/assessment_stream.mp4'
### Step 1: Understanding the Input Video ###
The first step is to understand the properties of the input videos before we can design a system to digest them. 
Use the `ffprobe` ([see documentation if needed](https://ffmpeg.org/ffprobe.html)) command line utility to obtain the `height`, `width`, and `frame rate` of the input video. We're also using the `-hide_banner` option to minimize the text output. 
​
**Instructions**: <br>
1.1 Execute the cell to preview the video. <br>
1.2 Execute the cell to gather information from input video stream. <br>
1.3 Modify the `<FIXME>`s _only_ to the correct values and execute the cell to mark your answer. _You can execute this cell multiple times until satisfactory_. <br>
# 1.1
# DO NOT CHANGE THIS CELL
from IPython.display import Video
Video(os.environ['TARGET_VIDEO_PATH_MP4'], width=720)
# 1.2
# DO NOT CHANGE THIS CELL
!ffprobe -i $TARGET_VIDEO_PATH \
         -hide_banner \
         2>&1| tee my_assessment/video_profile.txt 
Input #0, h264, from 'data/assessment_stream.h264':
  Duration: N/A, bitrate: N/A
    Stream #0:0: Video: h264 (High), yuv420p(progressive), 1280x720, 59.94 fps, 59.94 tbr, 1200k tbn, 119.88 tbc
# 1.3
FRAME_RATE=59.94
FRAME_HEIGHT=720
FRAME_WIDTH=1280
FRAME_CODEC='h264'
FRAME_COLOR_FORMAT='RGB'
​
# DO NOT CHANGE BELOW
Answer=f"""\
FRAME RATE: {round(FRAME_RATE)} FPS \
HEIGHT: {FRAME_HEIGHT} \
WIDTH: {FRAME_WIDTH} \
FRAME_CODEC: {FRAME_CODEC} \
FRAME_COLOR_FORMAT: {FRAME_COLOR_FORMAT} \
"""
​
!echo $Answer > my_assessment/answer_1.txt
### Step 2: Brainstorm AI Inference and Download a Pre-Trained Model ###
The next step is to brain storm the AI inference needed to achieve the objective. For this application, we need to detect cars in the frame and identify cases when the bounding box crosses below a threshold (illustrated below). 
​
<p><img src='images/tailgating_logic.png' width=720></p>
​
Fortunately, there is a [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/models/tlt_dashcamnet) purpose-built object detection model that has been trained on similar data as our video. We can use the [NGC CLI](https://ngc.nvidia.com/setup/installers/cli) to download the [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/models/tlt_dashcamnet) model for our application
**Instructions**: <br>
2.1 Execute the cell to install the NGC CLI. <br>
2.2 Execute the cell to use the `ngc registry mode list` command that lists all available models. We use the `--column name`, `--column repository`, and `--column application` options to display only the relevant columns. Afterwards, review the model card for [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/models/tlt_dashcamnet) and confirm that the model is fit for purpose. <br>
2.3 Update the `<FIXME>`s _only_ and execute the cell to download the [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/models/tlt_dashcamnet) model. The output of the command will generate a text file for grading purposes. _You can execute this cell multiple times until satisfactory_. <br>
# 2.1
# DO NOT CHANGE THIS CELL
import os
os.environ['NGC_DIR']='/dli/task/ngc_assets'
​
%env CLI=ngccli_cat_linux.zip
!mkdir -p $NGC_DIR/ngccli
!wget "https://ngc.nvidia.com/downloads/$CLI" -P $NGC_DIR/ngccli
!unzip -u "$NGC_DIR/ngccli/$CLI" \
       -d $NGC_DIR/ngccli/
!rm $NGC_DIR/ngccli/*.zip 
os.environ["PATH"]="{}/ngccli/ngc-cli:{}".format(os.getenv("NGC_DIR", ""), os.getenv("PATH", ""))
env: CLI=ngccli_cat_linux.zip
--2022-07-10 10:31:22--  https://ngc.nvidia.com/downloads/ngccli_cat_linux.zip
Resolving ngc.nvidia.com (ngc.nvidia.com)... 18.165.83.111, 18.165.83.119, 18.165.83.59, ...
Connecting to ngc.nvidia.com (ngc.nvidia.com)|18.165.83.111|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 33924412 (32M) [application/zip]
Saving to: ‘/dli/task/ngc_assets/ngccli/ngccli_cat_linux.zip’

ngccli_cat_linux.zi 100%[===================>]  32.35M  --.-KB/s    in 0.1s    

2022-07-10 10:31:22 (271 MB/s) - ‘/dli/task/ngc_assets/ngccli/ngccli_cat_linux.zip’ saved [33924412/33924412]

Archive:  /dli/task/ngc_assets/ngccli/ngccli_cat_linux.zip
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/yarl/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/yarl/_quoting_c.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libselinux.so.1  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/prettytable-2.0.0.dist-info/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/prettytable-2.0.0.dist-info/direct_url.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/prettytable-2.0.0.dist-info/top_level.txt  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/prettytable-2.0.0.dist-info/METADATA  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/prettytable-2.0.0.dist-info/COPYING  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/prettytable-2.0.0.dist-info/RECORD  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/prettytable-2.0.0.dist-info/WHEEL  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/prettytable-2.0.0.dist-info/INSTALLER  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/prettytable-2.0.0.dist-info/REQUESTED  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_queue.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_bisect.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/binascii.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/unicodedata.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_codecs_jp.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_ctypes.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_sha512.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_csv.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/pyexpat.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_ssl.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_sha256.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_socket.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/mmap.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_sha3.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/zlib.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_posixshmem.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/termios.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_codecs_hk.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_multibytecodec.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_struct.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/array.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_contextvars.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_decimal.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/fcntl.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/grp.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_codecs_tw.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_codecs_iso2022.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_multiprocessing.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/select.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_posixsubprocess.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/resource.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_opcode.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_codecs_cn.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_pickle.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_hashlib.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_json.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_datetime.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_sha1.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_md5.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_blake2.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_codecs_kr.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_random.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_elementtree.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_statistics.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_asyncio.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/math.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/lib-dynload/_heapq.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libfreebl3.so  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography/hazmat/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography/hazmat/bindings/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography/hazmat/bindings/_rust.abi3.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography/hazmat/bindings/_openssl.abi3.so  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/aiohttp/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/aiohttp/_http_parser.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/aiohttp/_websocket.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/aiohttp/_helpers.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/aiohttp/_http_writer.cpython-39-x86_64-linux-gnu.so  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/LICENSE  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/top_level.txt  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/METADATA  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/RECORD  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/WHEEL  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/LICENSE.APACHE  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/INSTALLER  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/LICENSE.BSD  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/cryptography-36.0.1.dist-info/LICENSE.PSF  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/jsonpickle-2.0.0.dist-info/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/jsonpickle-2.0.0.dist-info/direct_url.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/jsonpickle-2.0.0.dist-info/LICENSE  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/jsonpickle-2.0.0.dist-info/top_level.txt  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/jsonpickle-2.0.0.dist-info/METADATA  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/jsonpickle-2.0.0.dist-info/RECORD  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/jsonpickle-2.0.0.dist-info/WHEEL  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/jsonpickle-2.0.0.dist-info/INSTALLER  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/jsonpickle-2.0.0.dist-info/REQUESTED  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libgcc_s.so.1  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libcom_err.so.2  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libssl.so.10  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/frozenlist/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/frozenlist/_frozenlist.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libffi.so.6  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/grpc/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/grpc/_cython/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/grpc/_cython/cygrpc.cpython-39-x86_64-linux-gnu.so  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/grpc/_cython/_credentials/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/grpc/_cython/_credentials/roots.pem  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/google/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/google/protobuf/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/google/protobuf/internal/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/google/protobuf/internal/_api_implementation.cpython-39-x86_64-linux-gnu.so  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/google/protobuf/pyext/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/google/protobuf/pyext/_message.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libffi-9c61262e.so.8.1.0  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libz.so.1  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/_cffi_backend.cpython-39-x86_64-linux-gnu.so  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/cacert.pem  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudwatch/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudwatch/2010-08-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudwatch/2010-08-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudwatch/2010-08-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudwatch/2010-08-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudwatch/2010-08-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm/2014-11-06/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm/2014-11-06/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm/2014-11-06/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm/2014-11-06/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm/2014-11-06/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ivs/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ivs/2020-07-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ivs/2020-07-14/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ivs/2020-07-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/meteringmarketplace/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/meteringmarketplace/2016-01-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/meteringmarketplace/2016-01-14/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/meteringmarketplace/2016-01-14/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/meteringmarketplace/2016-01-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sns/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sns/2010-03-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sns/2010-03-31/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sns/2010-03-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sns/2010-03-31/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connectparticipant/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connectparticipant/2018-09-07/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connectparticipant/2018-09-07/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connectparticipant/2018-09-07/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediatailor/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediatailor/2018-04-23/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediatailor/2018-04-23/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediatailor/2018-04-23/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar-connections/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar-connections/2019-12-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar-connections/2019-12-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar-connections/2019-12-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amp/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amp/2020-08-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amp/2020-08-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amp/2020-08-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amp/2020-08-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sts/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sts/2011-06-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sts/2011-06-15/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sts/2011-06-15/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sts/2011-06-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-identity/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-identity/2014-06-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-identity/2014-06-30/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-identity/2014-06-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-identity/2014-06-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-media/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-media/2017-09-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-media/2017-09-30/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-media/2017-09-30/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-media/2017-09-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigatewaymanagementapi/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigatewaymanagementapi/2018-11-29/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigatewaymanagementapi/2018-11-29/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigatewaymanagementapi/2018-11-29/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift-data/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift-data/2019-12-20/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift-data/2019-12-20/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift-data/2019-12-20/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift-data/2019-12-20/paginators-1.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/transfer/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/transfer/2018-11-05/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/transfer/2018-11-05/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/transfer/2018-11-05/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecs/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecs/2014-11-13/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecs/2014-11-13/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecs/2014-11-13/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecs/2014-11-13/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecs/2014-11-13/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ram/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ram/2018-01-04/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ram/2018-01-04/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ram/2018-01-04/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr-containers/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr-containers/2020-10-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr-containers/2020-10-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr-containers/2020-10-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotfleethub/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotfleethub/2020-11-03/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotfleethub/2020-11-03/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotfleethub/2020-11-03/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resource-groups/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resource-groups/2017-11-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resource-groups/2017-11-27/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resource-groups/2017-11-27/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resource-groups/2017-11-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appstream/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appstream/2016-12-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appstream/2016-12-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appstream/2016-12-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appstream/2016-12-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appstream/2016-12-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/backup/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/backup/2018-11-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/backup/2018-11-15/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/backup/2018-11-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize-runtime/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize-runtime/2018-05-22/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize-runtime/2018-05-22/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize-runtime/2018-05-22/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint/2016-12-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint/2016-12-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint/2016-12-01/service-2.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/globalaccelerator/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/globalaccelerator/2018-08-08/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/globalaccelerator/2018-08-08/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/globalaccelerator/2018-08-08/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplacecommerceanalytics/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplacecommerceanalytics/2015-07-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplacecommerceanalytics/2015-07-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplacecommerceanalytics/2015-07-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplacecommerceanalytics/2015-07-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicecatalog-appregistry/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicecatalog-appregistry/2020-06-24/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicecatalog-appregistry/2020-06-24/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicecatalog-appregistry/2020-06-24/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sms/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sms/2016-10-24/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sms/2016-10-24/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sms/2016-10-24/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sms/2016-10-24/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling/2011-01-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling/2011-01-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling/2011-01-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling/2011-01-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mq/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mq/2017-11-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mq/2017-11-27/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mq/2017-11-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-models/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-models/2017-04-19/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-models/2017-04-19/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-models/2017-04-19/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-models/2017-04-19/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53domains/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53domains/2014-05-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53domains/2014-05-15/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53domains/2014-05-15/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53domains/2014-05-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/outposts/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/outposts/2019-12-03/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/outposts/2019-12-03/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/outposts/2019-12-03/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisanalyticsv2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisanalyticsv2/2018-05-23/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisanalyticsv2/2018-05-23/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisanalyticsv2/2018-05-23/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glue/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glue/2017-03-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glue/2017-03-31/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glue/2017-03-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glue/2017-03-31/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/budgets/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/budgets/2016-10-20/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/budgets/2016-10-20/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/budgets/2016-10-20/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/budgets/2016-10-20/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudformation/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudformation/2010-05-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudformation/2010-05-15/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudformation/2010-05-15/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudformation/2010-05-15/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudformation/2010-05-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm-contacts/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm-contacts/2021-05-03/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm-contacts/2021-05-03/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm-contacts/2021-05-03/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/swf/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/swf/2012-01-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/swf/2012-01-25/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/swf/2012-01-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/swf/2012-01-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodbstreams/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodbstreams/2012-08-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodbstreams/2012-08-10/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodbstreams/2012-08-10/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodbstreams/2012-08-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/2015-02-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/2015-02-02/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/2015-02-02/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/2015-02-02/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/2015-02-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/2014-09-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/2014-09-30/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/2014-09-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticache/2014-09-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr-public/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr-public/2020-10-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr-public/2020-10-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr-public/2020-10-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/shield/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/shield/2016-06-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/shield/2016-06-02/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/shield/2016-06-02/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/shield/2016-06-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3/2006-03-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3/2006-03-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3/2006-03-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3/2006-03-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3/2006-03-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/robomaker/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/robomaker/2018-06-29/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/robomaker/2018-06-29/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/robomaker/2018-06-29/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr/2009-03-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr/2009-03-31/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr/2009-03-31/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr/2009-03-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/emr/2009-03-31/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amplifybackend/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amplifybackend/2020-08-11/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amplifybackend/2020-08-11/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amplifybackend/2020-08-11/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot-data/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot-data/2015-05-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot-data/2015-05-28/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot-data/2015-05-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutvision/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutvision/2020-11-20/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutvision/2020-11-20/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutvision/2020-11-20/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/license-manager/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/license-manager/2018-08-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/license-manager/2018-08-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/license-manager/2018-08-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2-instance-connect/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2-instance-connect/2018-04-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2-instance-connect/2018-04-02/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2-instance-connect/2018-04-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/transcribe/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/transcribe/2017-10-26/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/transcribe/2017-10-26/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/transcribe/2017-10-26/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/transcribe/2017-10-26/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-cluster/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-cluster/2019-12-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-cluster/2019-12-02/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-cluster/2019-12-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workmailmessageflow/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workmailmessageflow/2019-05-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workmailmessageflow/2019-05-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workmailmessageflow/2019-05-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/logs/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/logs/2014-03-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/logs/2014-03-28/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/logs/2014-03-28/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/logs/2014-03-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amplify/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amplify/2017-07-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amplify/2017-07-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/amplify/2017-07-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connect-contact-lens/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connect-contact-lens/2020-08-21/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connect-contact-lens/2020-08-21/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connect-contact-lens/2020-08-21/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-archived-media/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-archived-media/2017-09-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-archived-media/2017-09-30/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-archived-media/2017-09-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-archived-media/2017-09-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm-incidents/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm-incidents/2018-05-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm-incidents/2018-05-10/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm-incidents/2018-05-10/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ssm-incidents/2018-05-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-runtime/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-runtime/2016-11-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-runtime/2016-11-28/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-runtime/2016-11-28/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lex-runtime/2016-11-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53/2013-04-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53/2013-04-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53/2013-04-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53/2013-04-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53/2013-04-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize/2018-05-22/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize/2018-05-22/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize/2018-05-22/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotwireless/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotwireless/2020-11-22/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotwireless/2020-11-22/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotwireless/2020-11-22/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/comprehendmedical/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/comprehendmedical/2018-10-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/comprehendmedical/2018-10-30/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/comprehendmedical/2018-10-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/application-insights/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/application-insights/2018-11-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/application-insights/2018-11-25/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/application-insights/2018-11-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/auditmanager/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/auditmanager/2017-07-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/auditmanager/2017-07-25/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/auditmanager/2017-07-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/worklink/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/worklink/2018-09-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/worklink/2018-09-25/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/worklink/2018-09-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/qldb/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/qldb/2019-01-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/qldb/2019-01-02/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/qldb/2019-01-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloud9/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloud9/2017-09-23/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloud9/2017-09-23/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloud9/2017-09-23/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloud9/2017-09-23/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/polly/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/polly/2016-06-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/polly/2016-06-10/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/polly/2016-06-10/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/polly/2016-06-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/batch/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/batch/2016-08-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/batch/2016-08-10/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/batch/2016-08-10/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/batch/2016-08-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appintegrations/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appintegrations/2020-07-29/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appintegrations/2020-07-29/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appintegrations/2020-07-29/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/timestream-write/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/timestream-write/2018-11-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/timestream-write/2018-11-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/timestream-write/2018-11-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dms/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dms/2016-01-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dms/2016-01-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dms/2016-01-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dms/2016-01-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dms/2016-01-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/account/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/account/2021-02-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/account/2021-02-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/account/2021-02-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisvideo/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisvideo/2017-09-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisvideo/2017-09-30/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisvideo/2017-09-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisvideo/2017-09-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticbeanstalk/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticbeanstalk/2010-12-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticbeanstalk/2010-12-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticbeanstalk/2010-12-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticbeanstalk/2010-12-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elasticbeanstalk/2010-12-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workspaces/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workspaces/2015-04-08/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workspaces/2015-04-08/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workspaces/2015-04-08/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workspaces/2015-04-08/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot1click-projects/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot1click-projects/2018-05-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot1click-projects/2018-05-14/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot1click-projects/2018-05-14/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot1click-projects/2018-05-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize-events/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize-events/2018-03-22/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize-events/2018-03-22/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/personalize-events/2018-03-22/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/memorydb/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/memorydb/2021-01-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/memorydb/2021-01-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/memorydb/2021-01-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/schemas/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/schemas/2019-12-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/schemas/2019-12-02/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/schemas/2019-12-02/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/schemas/2019-12-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iam/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iam/2010-05-08/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iam/2010-05-08/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iam/2010-05-08/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iam/2010-05-08/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iam/2010-05-08/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint-email/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint-email/2018-07-26/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint-email/2018-07-26/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint-email/2018-07-26/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/greengrassv2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/greengrassv2/2020-11-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/greengrassv2/2020-11-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/greengrassv2/2020-11-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appsync/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appsync/2017-07-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appsync/2017-07-25/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appsync/2017-07-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appsync/2017-07-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mobile/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mobile/2017-07-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mobile/2017-07-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mobile/2017-07-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mobile/2017-07-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/networkmanager/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/networkmanager/2019-07-05/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/networkmanager/2019-07-05/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/networkmanager/2019-07-05/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworkscm/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworkscm/2016-11-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworkscm/2016-11-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworkscm/2016-11-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworkscm/2016-11-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworkscm/2016-11-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/eks/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/eks/2017-11-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/eks/2017-11-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/eks/2017-11-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/eks/2017-11-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/eks/2017-11-01/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/eks/2017-11-01/service-2.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastictranscoder/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastictranscoder/2012-09-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastictranscoder/2012-09-25/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastictranscoder/2012-09-25/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastictranscoder/2012-09-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastictranscoder/2012-09-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm-pca/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm-pca/2017-08-22/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm-pca/2017-08-22/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm-pca/2017-08-22/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm-pca/2017-08-22/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm-pca/2017-08-22/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/nimble/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/nimble/2020-08-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/nimble/2020-08-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/nimble/2020-08-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/nimble/2020-08-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/greengrass/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/greengrass/2017-06-07/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/greengrass/2017-06-07/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/greengrass/2017-06-07/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/es/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/es/2015-01-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/es/2015-01-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/es/2015-01-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/es/2015-01-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-featurestore-runtime/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-featurestore-runtime/2020-07-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-featurestore-runtime/2020-07-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-featurestore-runtime/2020-07-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutmetrics/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutmetrics/2017-07-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutmetrics/2017-07-25/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutmetrics/2017-07-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeartifact/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeartifact/2018-09-22/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeartifact/2018-09-22/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeartifact/2018-09-22/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeartifact/2018-09-22/paginators-1.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/datapipeline/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/datapipeline/2012-10-29/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/datapipeline/2012-10-29/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/datapipeline/2012-10-29/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotsecuretunneling/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotsecuretunneling/2018-10-05/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotsecuretunneling/2018-10-05/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotsecuretunneling/2018-10-05/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudtrail/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudtrail/2013-11-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudtrail/2013-11-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudtrail/2013-11-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudtrail/2013-11-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplace-entitlement/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplace-entitlement/2017-01-11/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplace-entitlement/2017-01-11/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplace-entitlement/2017-01-11/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplace-entitlement/2017-01-11/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dax/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dax/2017-04-19/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dax/2017-04-19/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dax/2017-04-19/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dax/2017-04-19/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/comprehend/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/comprehend/2017-11-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/comprehend/2017-11-27/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/comprehend/2017-11-27/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/comprehend/2017-11-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workmail/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workmail/2017-10-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workmail/2017-10-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workmail/2017-10-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workmail/2017-10-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glacier/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glacier/2012-06-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glacier/2012-06-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glacier/2012-06-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glacier/2012-06-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/glacier/2012-06-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling-plans/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling-plans/2018-01-06/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling-plans/2018-01-06/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling-plans/2018-01-06/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/autoscaling-plans/2018-01-06/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lakeformation/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lakeformation/2017-03-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lakeformation/2017-03-31/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lakeformation/2017-03-31/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/datasync/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/datasync/2018-11-09/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/datasync/2018-11-09/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/datasync/2018-11-09/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-control-config/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-control-config/2020-11-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-control-config/2020-11-02/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-control-config/2020-11-02/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-control-config/2020-11-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotanalytics/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotanalytics/2017-11-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotanalytics/2017-11-27/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotanalytics/2017-11-27/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotanalytics/2017-11-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar/2017-04-19/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar/2017-04-19/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar/2017-04-19/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar/2017-04-19/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/machinelearning/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/machinelearning/2014-12-12/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/machinelearning/2014-12-12/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/machinelearning/2014-12-12/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/machinelearning/2014-12-12/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/machinelearning/2014-12-12/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/stepfunctions/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/stepfunctions/2016-11-23/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/stepfunctions/2016-11-23/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/stepfunctions/2016-11-23/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/stepfunctions/2016-11-23/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sms-voice/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sms-voice/2018-09-05/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sms-voice/2018-09-05/service-2.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediaconvert/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediaconvert/2017-08-29/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediaconvert/2017-08-29/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediaconvert/2017-08-29/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint-sms-voice/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint-sms-voice/2018-09-05/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pinpoint-sms-voice/2018-09-05/service-2.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/organizations/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/organizations/2016-11-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/organizations/2016-11-28/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/organizations/2016-11-28/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/organizations/2016-11-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/migrationhubstrategy/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/migrationhubstrategy/2020-02-19/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/migrationhubstrategy/2020-02-19/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/migrationhubstrategy/2020-02-19/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/migrationhubstrategy/2020-02-19/paginators-1.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appflow/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appflow/2020-08-23/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appflow/2020-08-23/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appflow/2020-08-23/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/location/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/location/2020-11-19/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/location/2020-11-19/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/location/2020-11-19/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf-regional/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf-regional/2016-11-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf-regional/2016-11-28/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf-regional/2016-11-28/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf-regional/2016-11-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplace-catalog/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplace-catalog/2018-09-17/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplace-catalog/2018-09-17/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/marketplace-catalog/2018-09-17/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elbv2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elbv2/2015-12-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elbv2/2015-12-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elbv2/2015-12-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elbv2/2015-12-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elbv2/2015-12-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/snowball/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/snowball/2016-06-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/snowball/2016-06-30/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/snowball/2016-06-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/snowball/2016-06-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/textract/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/textract/2018-06-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/textract/2018-06-27/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/textract/2018-06-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/grafana/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/grafana/2020-08-18/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/grafana/2020-08-18/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/grafana/2020-08-18/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediapackage-vod/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediapackage-vod/2018-11-07/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediapackage-vod/2018-11-07/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediapackage-vod/2018-11-07/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ds/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ds/2015-04-16/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ds/2015-04-16/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ds/2015-04-16/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ds/2015-04-16/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeguruprofiler/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeguruprofiler/2019-07-18/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeguruprofiler/2019-07-18/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeguruprofiler/2019-07-18/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fis/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fis/2020-12-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fis/2020-12-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fis/2020-12-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sdb/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sdb/2009-04-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sdb/2009-04-15/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sdb/2009-04-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudcontrol/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudcontrol/2021-09-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudcontrol/2021-09-30/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudcontrol/2021-09-30/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudcontrol/2021-09-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso-admin/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso-admin/2020-07-20/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso-admin/2020-07-20/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso-admin/2020-07-20/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dlm/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dlm/2018-01-12/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dlm/2018-01-12/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dlm/2018-01-12/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dlm/2018-01-12/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-runtime/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-runtime/2017-05-13/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-runtime/2017-05-13/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-runtime/2017-05-13/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-runtime/2017-05-13/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisanalytics/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisanalytics/2015-08-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisanalytics/2015-08-14/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisanalytics/2015-08-14/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesisanalytics/2015-08-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/timestream-query/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/timestream-query/2018-11-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/timestream-query/2018-11-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/timestream-query/2018-11-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf/2015-08-24/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf/2015-08-24/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf/2015-08-24/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/waf/2015-08-24/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/service-quotas/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/service-quotas/2019-06-24/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/service-quotas/2019-06-24/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/service-quotas/2019-06-24/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/importexport/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/importexport/2010-06-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/importexport/2010-06-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/importexport/2010-06-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotevents/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotevents/2018-07-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotevents/2018-07-27/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotevents/2018-07-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lambda/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lambda/2015-03-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lambda/2015-03-31/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lambda/2015-03-31/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lambda/2015-03-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lambda/2015-03-31/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lambda/2014-11-11/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lambda/2014-11-11/service-2.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elb/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elb/2012-06-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elb/2012-06-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elb/2012-06-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elb/2012-06-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elb/2012-06-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/finspace-data/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/finspace-data/2020-07-13/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/finspace-data/2020-07-13/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/finspace-data/2020-07-13/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/panorama/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/panorama/2019-07-24/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/panorama/2019-07-24/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/panorama/2019-07-24/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/qldb-session/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/qldb-session/2019-07-11/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/qldb-session/2019-07-11/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/qldb-session/2019-07-11/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/neptune/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/neptune/2014-10-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/neptune/2014-10-31/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/neptune/2014-10-31/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/neptune/2014-10-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/neptune/2014-10-31/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/neptune/2014-10-31/service-2.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-messaging/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-messaging/2021-05-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-messaging/2021-05-15/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-messaging/2021-05-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connect/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connect/2017-08-08/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connect/2017-08-08/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connect/2017-08-08/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/connect/2017-08-08/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pi/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pi/2018-02-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pi/2018-02-27/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pi/2018-02-27/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pi/2018-02-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lexv2-runtime/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lexv2-runtime/2020-08-07/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lexv2-runtime/2020-08-07/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lexv2-runtime/2020-08-07/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/support/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/support/2013-04-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/support/2013-04-15/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/support/2013-04-15/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/support/2013-04-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore-data/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore-data/2017-09-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore-data/2017-09-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore-data/2017-09-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore-data/2017-09-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/managedblockchain/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/managedblockchain/2018-09-24/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/managedblockchain/2018-09-24/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/managedblockchain/2018-09-24/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/endpoints.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/macie2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/macie2/2020-01-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/macie2/2020-01-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/macie2/2020-01-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/guardduty/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/guardduty/2017-11-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/guardduty/2017-11-28/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/guardduty/2017-11-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ses/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ses/2010-12-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ses/2010-12-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ses/2010-12-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ses/2010-12-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ses/2010-12-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/frauddetector/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/frauddetector/2019-11-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/frauddetector/2019-11-15/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/frauddetector/2019-11-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot-jobs-data/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot-jobs-data/2017-09-29/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot-jobs-data/2017-09-29/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot-jobs-data/2017-09-29/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot-jobs-data/2017-09-29/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ebs/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ebs/2019-11-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ebs/2019-11-02/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ebs/2019-11-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-meetings/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-meetings/2021-07-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-meetings/2021-07-15/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-meetings/2021-07-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/docdb/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/docdb/2014-10-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/docdb/2014-10-31/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/docdb/2014-10-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/docdb/2014-10-31/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/docdb/2014-10-31/service-2.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-idp/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-idp/2016-04-18/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-idp/2016-04-18/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-idp/2016-04-18/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-idp/2016-04-18/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-10-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-10-31/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-10-31/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-10-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-10-31/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-10-31/service-2.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-09-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-09-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-09-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds/2014-09-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/medialive/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/medialive/2017-10-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/medialive/2017-10-14/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/medialive/2017-10-14/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/medialive/2017-10-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opensearch/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opensearch/2021-01-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opensearch/2021-01-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opensearch/2021-01-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearchdomain/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearchdomain/2013-01-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearchdomain/2013-01-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearchdomain/2013-01-01/service-2.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ce/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ce/2017-10-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ce/2017-10-25/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ce/2017-10-25/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ce/2017-10-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lightsail/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lightsail/2016-11-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lightsail/2016-11-28/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lightsail/2016-11-28/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lightsail/2016-11-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/directconnect/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/directconnect/2012-10-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/directconnect/2012-10-25/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/directconnect/2012-10-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/directconnect/2012-10-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearch/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearch/2011-02-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearch/2011-02-01/service-2.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearch/2013-01-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearch/2013-01-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudsearch/2013-01-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/translate/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/translate/2017-07-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/translate/2017-07-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/translate/2017-07-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/translate/2017-07-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediaconnect/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediaconnect/2018-11-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediaconnect/2018-11-14/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediaconnect/2018-11-14/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediaconnect/2018-11-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/savingsplans/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/savingsplans/2019-06-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/savingsplans/2019-06-28/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/savingsplans/2019-06-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediapackage/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediapackage/2017-10-12/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediapackage/2017-10-12/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediapackage/2017-10-12/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/signer/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/signer/2017-08-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/signer/2017-08-25/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/signer/2017-08-25/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/signer/2017-08-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/signer/2017-08-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/forecastquery/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/forecastquery/2018-06-26/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/forecastquery/2018-06-26/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/forecastquery/2018-06-26/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/applicationcostprofiler/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/applicationcostprofiler/2020-09-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/applicationcostprofiler/2020-09-10/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/applicationcostprofiler/2020-09-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appmesh/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appmesh/2019-01-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appmesh/2019-01-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appmesh/2019-01-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appmesh/2018-10-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appmesh/2018-10-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appmesh/2018-10-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wisdom/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wisdom/2020-10-19/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wisdom/2020-10-19/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wisdom/2020-10-19/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicecatalog/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicecatalog/2015-12-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicecatalog/2015-12-10/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicecatalog/2015-12-10/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicecatalog/2015-12-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/databrew/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/databrew/2017-07-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/databrew/2017-07-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/databrew/2017-07-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mgn/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mgn/2020-02-26/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mgn/2020-02-26/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mgn/2020-02-26/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apprunner/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apprunner/2020-05-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apprunner/2020-05-15/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apprunner/2020-05-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workdocs/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workdocs/2016-05-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workdocs/2016-05-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workdocs/2016-05-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/workdocs/2016-05-01/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/_retry.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/identitystore/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/identitystore/2020-06-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/identitystore/2020-06-15/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/identitystore/2020-06-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resiliencehub/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resiliencehub/2020-04-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resiliencehub/2020-04-30/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resiliencehub/2020-04-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sqs/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sqs/2012-11-05/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sqs/2012-11-05/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sqs/2012-11-05/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sqs/2012-11-05/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/synthetics/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/synthetics/2017-10-11/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/synthetics/2017-10-11/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/synthetics/2017-10-11/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wafv2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wafv2/2019-07-29/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wafv2/2019-07-29/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wafv2/2019-07-29/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodb/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodb/2012-08-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodb/2012-08-10/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodb/2012-08-10/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodb/2012-08-10/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dynamodb/2012-08-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/storagegateway/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/storagegateway/2013-06-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/storagegateway/2013-06-30/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/storagegateway/2013-06-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/storagegateway/2013-06-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mgh/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mgh/2017-05-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mgh/2017-05-31/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mgh/2017-05-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mgh/2017-05-31/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot1click-devices/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot1click-devices/2018-05-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot1click-devices/2018-05-14/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot1click-devices/2018-05-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/application-autoscaling/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/application-autoscaling/2016-02-06/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/application-autoscaling/2016-02-06/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/application-autoscaling/2016-02-06/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/application-autoscaling/2016-02-06/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kms/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kms/2014-11-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kms/2014-11-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kms/2014-11-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kms/2014-11-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/groundstation/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/groundstation/2019-05-23/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/groundstation/2019-05-23/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/groundstation/2019-05-23/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds-data/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds-data/2018-08-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds-data/2018-08-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rds-data/2018-08-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicediscovery/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicediscovery/2017-03-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicediscovery/2017-03-14/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicediscovery/2017-03-14/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/servicediscovery/2017-03-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/detective/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/detective/2018-10-26/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/detective/2018-10-26/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/detective/2018-10-26/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotdeviceadvisor/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotdeviceadvisor/2020-09-18/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotdeviceadvisor/2020-09-18/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotdeviceadvisor/2020-09-18/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore/2017-09-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore/2017-09-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore/2017-09-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mediastore/2017-09-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm/2015-12-08/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm/2015-12-08/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm/2015-12-08/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm/2015-12-08/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/acm/2015-12-08/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/events/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/events/2015-10-07/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/events/2015-10-07/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/events/2015-10-07/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/events/2015-10-07/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/events/2014-02-03/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/events/2014-02-03/service-2.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kafka/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kafka/2018-11-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kafka/2018-11-14/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kafka/2018-11-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotthingsgraph/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotthingsgraph/2018-09-06/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotthingsgraph/2018-09-06/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotthingsgraph/2018-09-06/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3control/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3control/2018-08-20/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3control/2018-08-20/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3control/2018-08-20/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codecommit/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codecommit/2015-04-13/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codecommit/2015-04-13/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codecommit/2015-04-13/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codecommit/2015-04-13/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pricing/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pricing/2017-10-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pricing/2017-10-15/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pricing/2017-10-15/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/pricing/2017-10-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/voice-id/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/voice-id/2021-09-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/voice-id/2021-09-27/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/voice-id/2021-09-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/quicksight/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/quicksight/2018-04-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/quicksight/2018-04-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/quicksight/2018-04-01/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/quicksight/2018-04-01/paginators-1.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resourcegroupstaggingapi/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resourcegroupstaggingapi/2017-01-26/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resourcegroupstaggingapi/2017-01-26/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resourcegroupstaggingapi/2017-01-26/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/resourcegroupstaggingapi/2017-01-26/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworks/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworks/2013-02-18/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworks/2013-02-18/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworks/2013-02-18/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworks/2013-02-18/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/opsworks/2013-02-18/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codepipeline/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codepipeline/2015-07-09/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codepipeline/2015-07-09/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codepipeline/2015-07-09/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codepipeline/2015-07-09/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sesv2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sesv2/2019-09-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sesv2/2019-09-27/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sesv2/2019-09-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso/2019-06-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso/2019-06-10/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso/2019-06-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-readiness/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-readiness/2019-12-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-readiness/2019-12-02/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53-recovery-readiness/2019-12-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53resolver/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53resolver/2018-04-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53resolver/2018-04-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53resolver/2018-04-01/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/route53resolver/2018-04-01/paginators-1.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rekognition/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rekognition/2016-06-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rekognition/2016-06-27/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rekognition/2016-06-27/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rekognition/2016-06-27/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/rekognition/2016-06-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/xray/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/xray/2016-04-12/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/xray/2016-04-12/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/xray/2016-04-12/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/xray/2016-04-12/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-signaling/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-signaling/2019-12-04/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-signaling/2019-12-04/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis-video-signaling/2019-12-04/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/firehose/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/firehose/2015-08-04/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/firehose/2015-08-04/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/firehose/2015-08-04/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/firehose/2015-08-04/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-sync/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-sync/2014-06-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-sync/2014-06-30/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cognito-sync/2014-06-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fms/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fms/2018-01-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fms/2018-01-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fms/2018-01-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fms/2018-01-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kafkaconnect/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kafkaconnect/2021-09-14/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kafkaconnect/2021-09-14/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kafkaconnect/2021-09-14/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cur/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cur/2017-01-06/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cur/2017-01-06/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cur/2017-01-06/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cur/2017-01-06/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar-notifications/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar-notifications/2019-10-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar-notifications/2019-10-15/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codestar-notifications/2019-10-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/devops-guru/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/devops-guru/2020-12-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/devops-guru/2020-12-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/devops-guru/2020-12-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigateway/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigateway/2015-07-09/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigateway/2015-07-09/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigateway/2015-07-09/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigateway/2015-07-09/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mwaa/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mwaa/2020-07-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mwaa/2020-07-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mwaa/2020-07-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/proton/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/proton/2020-07-20/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/proton/2020-07-20/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/proton/2020-07-20/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/proton/2020-07-20/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/devicefarm/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/devicefarm/2015-06-23/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/devicefarm/2015-06-23/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/devicefarm/2015-06-23/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/devicefarm/2015-06-23/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutequipment/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutequipment/2020-12-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutequipment/2020-12-15/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lookoutequipment/2020-12-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-edge/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-edge/2020-09-23/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-edge/2020-09-23/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-edge/2020-09-23/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotevents-data/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotevents-data/2018-10-23/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotevents-data/2018-10-23/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotevents-data/2018-10-23/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsmv2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsmv2/2017-04-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsmv2/2017-04-28/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsmv2/2017-04-28/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsmv2/2017-04-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsm/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsm/2014-05-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsm/2014-05-30/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsm/2014-05-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudhsm/2014-05-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3outposts/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3outposts/2017-07-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3outposts/2017-07-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/s3outposts/2017-07-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fsx/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fsx/2018-03-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fsx/2018-03-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/fsx/2018-03-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/inspector/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/inspector/2016-02-16/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/inspector/2016-02-16/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/inspector/2016-02-16/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/inspector/2016-02-16/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/inspector/2015-08-18/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/inspector/2015-08-18/service-2.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/secretsmanager/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/secretsmanager/2017-10-17/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/secretsmanager/2017-10-17/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/secretsmanager/2017-10-17/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/secretsmanager/2017-10-17/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/secretsmanager/2017-10-17/service-2.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr/2015-09-21/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr/2015-09-21/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr/2015-09-21/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr/2015-09-21/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ecr/2015-09-21/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/gamelift/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/gamelift/2015-10-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/gamelift/2015-10-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/gamelift/2015-10-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/gamelift/2015-10-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/network-firewall/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/network-firewall/2020-11-12/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/network-firewall/2020-11-12/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/network-firewall/2020-11-12/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/migrationhub-config/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/migrationhub-config/2019-06-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/migrationhub-config/2019-06-30/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/migrationhub-config/2019-06-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/finspace/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/finspace/2021-03-12/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/finspace/2021-03-12/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/finspace/2021-03-12/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/snow-device-management/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/snow-device-management/2021-08-04/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/snow-device-management/2021-08-04/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/snow-device-management/2021-08-04/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/imagebuilder/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/imagebuilder/2019-12-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/imagebuilder/2019-12-02/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/imagebuilder/2019-12-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-09-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-09-15/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-09-15/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-09-15/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-09-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-03-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-03-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-03-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-03-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-04-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-04-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-04-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-04-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2014-09-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2014-09-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2014-09-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2014-09-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2014-10-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2014-10-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2014-10-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2014-10-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-10-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-10-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-10-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-10-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-11-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-11-15/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-11-15/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-11-15/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2016-11-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-04-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-04-15/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-04-15/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/ec2/2015-04-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codedeploy/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codedeploy/2014-10-06/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codedeploy/2014-10-06/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codedeploy/2014-10-06/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codedeploy/2014-10-06/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codedeploy/2014-10-06/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/compute-optimizer/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/compute-optimizer/2019-11-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/compute-optimizer/2019-11-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/compute-optimizer/2019-11-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/health/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/health/2016-08-04/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/health/2016-08-04/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/health/2016-08-04/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/health/2016-08-04/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigatewayv2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigatewayv2/2018-11-29/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigatewayv2/2018-11-29/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/apigatewayv2/2018-11-29/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastic-inference/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastic-inference/2017-07-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastic-inference/2017-07-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/elastic-inference/2017-07-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot/2015-05-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot/2015-05-28/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot/2015-05-28/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iot/2015-05-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime/2018-05-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime/2018-05-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime/2018-05-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appconfig/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appconfig/2019-10-09/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appconfig/2019-10-09/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/appconfig/2019-10-09/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/macie/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/macie/2017-12-19/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/macie/2017-12-19/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/macie/2017-12-19/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/macie/2017-12-19/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mturk/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mturk/2017-01-17/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mturk/2017-01-17/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mturk/2017-01-17/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/mturk/2017-01-17/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/serverlessrepo/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/serverlessrepo/2017-09-08/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/serverlessrepo/2017-09-08/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/serverlessrepo/2017-09-08/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-01-28/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-01-28/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-01-28/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-01-28/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-05-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-05-31/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-05-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-05-31/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-11-05/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-11-05/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-11-05/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-11-05/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-11-05/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-09-17/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-09-17/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-09-17/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-09-17/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-08-20/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-08-20/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-08-20/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-08-20/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-03-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-03-25/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-03-25/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-03-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-03-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-04-17/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-04-17/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-04-17/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-04-17/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-10-21/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-10-21/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-10-21/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-10-21/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2019-03-26/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2019-03-26/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2019-03-26/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2019-03-26/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2019-03-26/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-08-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-08-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-08-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-08-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-11-06/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-11-06/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-11-06/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2014-11-06/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-07-27/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-07-27/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-07-27/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2015-07-27/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-06-18/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-06-18/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-06-18/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-06-18/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2018-06-18/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-01-13/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-01-13/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-01-13/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-01-13/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-10-30/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-10-30/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-10-30/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-10-30/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2017-10-30/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-09-29/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-09-29/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-09-29/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-09-29/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-11-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-11-25/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-11-25/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-11-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-11-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2020-05-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2020-05-31/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2020-05-31/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2020-05-31/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2020-05-31/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-09-07/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-09-07/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-09-07/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/cloudfront/2016-09-07/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kendra/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kendra/2019-02-03/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kendra/2019-02-03/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kendra/2019-02-03/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso-oidc/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso-oidc/2019-06-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso-oidc/2019-06-10/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sso-oidc/2019-06-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/discovery/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/discovery/2015-11-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/discovery/2015-11-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/discovery/2015-11-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/discovery/2015-11-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/healthlake/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/healthlake/2017-07-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/healthlake/2017-07-01/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/healthlake/2017-07-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/accessanalyzer/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/accessanalyzer/2019-11-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/accessanalyzer/2019-11-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/accessanalyzer/2019-11-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wellarchitected/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wellarchitected/2020-03-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wellarchitected/2020-03-31/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/wellarchitected/2020-03-31/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift/2012-12-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift/2012-12-01/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift/2012-12-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift/2012-12-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/redshift/2012-12-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dataexchange/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dataexchange/2017-07-25/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dataexchange/2017-07-25/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/dataexchange/2017-07-25/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/alexaforbusiness/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/alexaforbusiness/2017-11-09/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/alexaforbusiness/2017-11-09/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/alexaforbusiness/2017-11-09/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/alexaforbusiness/2017-11-09/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-a2i-runtime/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-a2i-runtime/2019-11-07/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-a2i-runtime/2019-11-07/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker-a2i-runtime/2019-11-07/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/clouddirectory/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/clouddirectory/2016-05-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/clouddirectory/2016-05-10/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/clouddirectory/2016-05-10/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/clouddirectory/2017-01-11/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/clouddirectory/2017-01-11/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/clouddirectory/2017-01-11/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/clouddirectory/2017-01-11/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/honeycode/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/honeycode/2020-03-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/honeycode/2020-03-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/honeycode/2020-03-01/paginators-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/honeycode/2020-03-01/paginators-1.sdk-extras.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/config/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/config/2014-11-12/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/config/2014-11-12/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/config/2014-11-12/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/config/2014-11-12/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codebuild/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codebuild/2016-10-06/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codebuild/2016-10-06/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codebuild/2016-10-06/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codebuild/2016-10-06/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lexv2-models/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lexv2-models/2020-08-07/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lexv2-models/2020-08-07/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lexv2-models/2020-08-07/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/lexv2-models/2020-08-07/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis/2013-12-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis/2013-12-02/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis/2013-12-02/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis/2013-12-02/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/kinesis/2013-12-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeguru-reviewer/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeguru-reviewer/2019-09-19/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeguru-reviewer/2019-09-19/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeguru-reviewer/2019-09-19/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/codeguru-reviewer/2019-09-19/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/efs/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/efs/2015-02-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/efs/2015-02-01/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/efs/2015-02-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/efs/2015-02-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/athena/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/athena/2017-05-18/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/athena/2017-05-18/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/athena/2017-05-18/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/athena/2017-05-18/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/securityhub/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/securityhub/2018-10-26/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/securityhub/2018-10-26/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/securityhub/2018-10-26/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/braket/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/braket/2019-09-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/braket/2019-09-01/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/braket/2019-09-01/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-identity/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-identity/2021-04-20/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-identity/2021-04-20/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/chime-sdk-identity/2021-04-20/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker/2017-07-24/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker/2017-07-24/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker/2017-07-24/examples-1.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker/2017-07-24/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/sagemaker/2017-07-24/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotsitewise/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotsitewise/2019-12-02/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotsitewise/2019-12-02/waiters-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotsitewise/2019-12-02/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/iotsitewise/2019-12-02/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/forecast/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/forecast/2018-06-26/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/forecast/2018-06-26/service-2.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/forecast/2018-06-26/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/customer-profiles/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/customer-profiles/2020-08-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/customer-profiles/2020-08-15/service-2.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/botocore/data/customer-profiles/2020-08-15/paginators-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/psutil/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/psutil/_psutil_linux.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/psutil/_psutil_posix.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libpcre.so.1  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/examples/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/examples/cloudfront.rst  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/examples/s3.rst  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/cloudwatch/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/cloudwatch/2010-08-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/cloudwatch/2010-08-01/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/sns/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/sns/2010-03-31/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/sns/2010-03-31/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/cloudformation/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/cloudformation/2010-05-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/cloudformation/2010-05-15/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/s3/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/s3/2006-03-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/s3/2006-03-01/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/iam/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/iam/2010-05-08/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/iam/2010-05-08/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/glacier/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/glacier/2012-06-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/glacier/2012-06-01/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/sqs/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/sqs/2012-11-05/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/sqs/2012-11-05/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/dynamodb/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/dynamodb/2012-08-10/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/dynamodb/2012-08-10/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/opsworks/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/opsworks/2013-02-18/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/opsworks/2013-02-18/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2016-09-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2016-09-15/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2015-03-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2015-03-01/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2016-04-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2016-04-01/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2014-10-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2014-10-01/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2015-10-01/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2015-10-01/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2016-11-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2016-11-15/resources-1.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2015-04-15/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/boto3/data/ec2/2015-04-15/resources-1.json  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/base_library.zip  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libgssapi_krb5.so.2  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libkrb5support.so.0  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libkrb5.so.3  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libk5crypto.so.3  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libcrypto.so.10  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/ngc  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/wcwidth/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/wcwidth/version.json  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/thrift/
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/thrift/protocol/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/thrift/protocol/fastbinary.cpython-39-x86_64-linux-gnu.so  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/multidict/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/multidict/_multidict.cpython-39-x86_64-linux-gnu.so  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libkeyutils.so.1  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libstdc++.so.6  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/certifi/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/certifi/cacert.pem  
   creating: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/direct_url.json  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/LICENSE  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/top_level.txt  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/namespace_packages.txt  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/METADATA  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/RECORD  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/WHEEL  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/INSTALLER  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli/google_api_core-2.2.2.dist-info/REQUESTED  
  inflating: /dli/task/ngc_assets/ngccli/ngc-cli/libpython3.9.so.1.0  
 extracting: /dli/task/ngc_assets/ngccli/ngc-cli.md5  
# 2.2
# DO NOT CHANGE THIS CELL
!ngc registry model list nvidia/tao/* --column name --column repository --column application
[{
    "application": "Object Detection",
    "createdDate": "2021-08-16T15:03:41.786Z",
    "description": "3 class object detection network to detect people in an image.",
    "displayName": "PeopleNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Object Detection",
                "Smart City",
                "People Detection",
                "TAO",
                "Public Safety",
                "Smart Infrastructure",
                "Robotics",
                "Retail",
                "CV",
                "Metropolis",
                "TAO Toolkit",
                "DetectNet_v2",
                "Healthcare",
                "Computer Vision"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_quantized_v2.6",
    "latestVersionSizeInBytes": 89162881,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "peoplenet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-26T20:07:00.247Z"
},{
    "application": "Object Detection",
    "createdDate": "2021-08-16T15:03:41.798Z",
    "description": "1 class object detection network to detect faces in an image.",
    "displayName": "FaceDetectIR",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Object Detection",
                "Smart City",
                "TAO",
                "Public Safety",
                "IR",
                "Smart Infrastructure",
                "Image Classification",
                "Retail",
                "CV",
                "Metropolis",
                "TAO Toolkit",
                "Healthcare",
                "DetectNet_v2"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "pruned_v1.0.1",
    "latestVersionSizeInBytes": 9532530,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "facedetectir",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2021-11-08T05:26:22.109Z"
},{
    "application": "Classification",
    "createdDate": "2021-08-16T15:03:41.810Z",
    "description": "Resnet18 model to classify a car crop into 1 out 20 car brands.",
    "displayName": "VehicleMakeNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Image Classification",
                "Smart City",
                "TAO",
                "CV",
                "Metropolis",
                "Traffic",
                "Public Safety",
                "TAO Toolkit",
                "Smart Infrastructure",
                "Computer Vision",
                "Vehicle Classification"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "pruned_v1.0.1",
    "latestVersionSizeInBytes": 17247772,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "vehiclemakenet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2021-11-08T04:32:26.789Z"
},{
    "application": "Object Detection",
    "createdDate": "2021-08-16T15:03:42.130Z",
    "description": "4 class object detection network to detect cars in an image.",
    "displayName": "DashCamNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "CV",
                "Metropolis",
                "TAO",
                "Transfer Learning",
                "TAO Toolkit",
                "AI",
                "Smart Infrastructure",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "pruned_v1.0.2",
    "latestVersionSizeInBytes": 6967865,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "dashcamnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-06-16T15:42:54.950Z"
},{
    "application": "Object Detection",
    "createdDate": "2021-08-16T15:03:42.144Z",
    "description": "Object Detection network to detect license plates in an image of a car.",
    "displayName": "LPDNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Object Detection",
                "Smart City",
                "TAO",
                "Traffic",
                "Public Safety",
                "License plate detection",
                "Smart Infrastructure",
                "CV",
                "Metropolis",
                "TAO Toolkit",
                "DetectNet_v2",
                "Computer Vision"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "pruned_v2.1",
    "latestVersionSizeInBytes": 3975758,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "lpdnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-25T15:40:21.041Z"
},{
    "application": "Instance Segmentation",
    "createdDate": "2021-08-16T15:03:42.177Z",
    "description": "1 class instance segmentation network to detect and segment instances of people in an image.",
    "displayName": "PeopleSegNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "MaskRCNN",
                "TAO",
                "Public Safety",
                "Smart Infrastructure",
                "Robotics",
                "Retail",
                "CV",
                "Metropolis",
                "Instance Segmentation",
                "TAO Toolkit",
                "Healthcare",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v2.0.2",
    "latestVersionSizeInBytes": 73969636,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "peoplesegnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "teamName": "tao",
    "updatedDate": "2022-05-25T15:52:00.785Z"
},{
    "application": "Classification",
    "createdDate": "2021-08-16T15:03:42.190Z",
    "description": "Resnet18 model to classify a car crop into 1 out 6 car types.",
    "displayName": "VehicleTypeNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Image Classification",
                "Object Detection",
                "Smart City",
                "TAO",
                "CV",
                "Metropolis",
                "Traffic",
                "Public Safety",
                "TAO Toolkit",
                "Smart Infrastructure",
                "Vehicle Classification"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "pruned_v1.0.1",
    "latestVersionSizeInBytes": 19980344,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "vehicletypenet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2021-11-08T04:58:38.199Z"
},{
    "application": "Object Detection",
    "createdDate": "2021-08-16T15:03:42.235Z",
    "description": "4 class object detection network to detect cars in an image.",
    "displayName": "TrafficCamNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "CV",
                "Metropolis",
                "TAO",
                "Transfer Learning",
                "TAO Toolkit",
                "AI",
                "Smart Infrastructure",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "pruned_v1.0.2",
    "latestVersionSizeInBytes": 5453692,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "trafficcamnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-25T21:59:56.894Z"
},{
    "application": "Character Recognition",
    "createdDate": "2021-08-16T15:03:42.268Z",
    "description": "Model to recognize characters from the image crop of a License Plate.",
    "displayName": "License Plate Recognition",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "TAO",
                "CV",
                "Metropolis",
                "Traffic",
                "Public Safety",
                "TAO Toolkit",
                "License Plate recognition",
                "Smart Infrastructure",
                "Computer Vision",
                "OCR"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v1.0",
    "latestVersionSizeInBytes": 231797006,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "lprnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2021-11-08T05:39:22.035Z"
},{
    "application": "Classification",
    "createdDate": "2021-08-16T15:53:38.509Z",
    "description": "Pretrained weights to facilitate transfer learning using TAO Toolkit.",
    "displayName": "TAO Pretrained Classification",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "TAO",
                "Industrial",
                "Inspection",
                "Public Safety",
                "EfficientNet",
                "Smart Infrastructure",
                "ResNet",
                "Retail",
                "CV",
                "Metropolis",
                "VGG",
                "TAO Toolkit",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        }
    ],
    "latestVersionIdStr": "cspdarknet_tiny",
    "latestVersionSizeInBytes": 29955696,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "pretrained_classification",
    "orgName": "nvidia",
    "precision": "FP32",
    "teamName": "tao",
    "updatedDate": "2021-11-23T07:41:04.189Z"
},{
    "application": "Object Detection",
    "createdDate": "2021-08-16T15:53:38.516Z",
    "description": "Pretrained weights to facilitate transfer learning using TAO Toolkit.",
    "displayName": "TAO Pretrained Object Detection",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DSSD",
                "DeepStream",
                "Smart City",
                "TAO",
                "Industrial",
                "SSD",
                "Inspection",
                "Public Safety",
                "EfficientNet",
                "Smart Infrastructure",
                "ResNet",
                "YOLO",
                "Retail",
                "CV",
                "Metropolis",
                "TAO Toolkit",
                "FasterRCNN",
                "TLT",
                "RetinaNet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        }
    ],
    "latestVersionIdStr": "cspdarknet_tiny",
    "latestVersionSizeInBytes": 29955696,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "pretrained_object_detection",
    "orgName": "nvidia",
    "precision": "FP32",
    "teamName": "tao",
    "updatedDate": "2022-06-03T23:37:04.905Z"
},{
    "application": "Object Detection",
    "createdDate": "2021-08-16T15:53:38.600Z",
    "description": "Pretrained weights to facilitate transfer learning using TAO Toolkit.",
    "displayName": "TAO Pretrained DetectNet V2",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Retail",
                "Smart City",
                "TAO",
                "CV",
                "Metropolis",
                "Industrial",
                "Inspection",
                "Public Safety",
                "TAO Toolkit",
                "DetectNet_v2",
                "Smart Infrastructure"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "resnet34",
    "latestVersionSizeInBytes": 178944632,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "pretrained_detectnet_v2",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-02-02T17:18:27.337Z"
},{
    "application": "Semantic Segmentation",
    "createdDate": "2021-08-16T16:34:42.315Z",
    "description": "Pretrained weights to facilitate transfer learning using Transfer Learning Toolkit.",
    "displayName": "TAO Pretrained Semantic Segmentation",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "TAO",
                "Industrial",
                "Inspection",
                "Public Safety",
                "Smart Infrastructure",
                "Robotics",
                "UNet",
                "Retail",
                "CV",
                "Metropolis",
                "TAO Toolkit",
                "Semantic Segmentation",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "vgg19",
    "latestVersionSizeInBytes": 161183816,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "pretrained_semantic_segmentation",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2021-09-20T21:30:44.387Z"
},{
    "application": "Instance Segmentation",
    "createdDate": "2021-08-16T16:34:42.327Z",
    "description": "Pretrained weights to facilitate transfer learning using TAO Toolkit.",
    "displayName": "TAO Pretrained Instance Segmentation",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "TAO",
                "MaskRCNN",
                "Inspection",
                "Public Safety",
                "Smart Infrastructure",
                "Robotics",
                "Retail",
                "Metropolis",
                "CV",
                "Instance Segmentation",
                "TAO Toolkit",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        }
    ],
    "latestVersionIdStr": "resnet10",
    "latestVersionSizeInBytes": 40175904,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "pretrained_instance_segmentation",
    "orgName": "nvidia",
    "precision": "FP32",
    "teamName": "tao",
    "updatedDate": "2021-09-20T21:30:44.113Z"
},{
    "application": "Speech to Text",
    "createdDate": "2021-08-18T20:04:57.047Z",
    "description": "Speech to Text Jasper model for English.",
    "displayName": "Speech to Text English Jasper",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "ASR",
                "English",
                "Jasper",
                "TAO",
                "Riva",
                "TAO Toolkit"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.2",
    "latestVersionSizeInBytes": 1234711099,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "speechtotext_english_jasper",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-03-26T03:23:18.943Z"
},{
    "application": "Domain Classification",
    "createdDate": "2021-08-18T20:04:57.163Z",
    "description": "Domain classification of the query for weather chat bot.",
    "displayName": "Domain Classification English Bert",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "NLP",
                "TAO",
                "Riva",
                "Domain Classification",
                "TAO Toolkit",
                "BERT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 440794733,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "domainclassification_english_bert",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:43:45.183Z"
},{
    "application": "Named Entity Recognition",
    "createdDate": "2021-08-18T20:04:57.311Z",
    "description": "The model identifies a category/entity the word in the input text belongs to.",
    "displayName": "Named Entity Recognition Bert",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "Named Entity Recognition",
                "NLP",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "BERT",
                "NER"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 440857990,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "namedentityrecognition_english_bert",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:43:41.863Z"
},{
    "application": "Speech to Text",
    "createdDate": "2021-08-18T20:04:57.753Z",
    "description": "Speech to Text Citrinet models for English.",
    "displayName": "Speech to Text English Citrinet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "ASR",
                "English",
                "STT",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Citrinet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v3.0",
    "latestVersionSizeInBytes": 566724116,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "speechtotext_english_citrinet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-03-26T03:24:49.050Z"
},{
    "application": "Joint Intent and Slot classification",
    "createdDate": "2021-08-18T20:04:58.439Z",
    "description": "Intent and Slot classification of the qeuries for the weather chat bot (trained on weather chat bot data).",
    "displayName": "Joint Intent and Slot Classification Bert",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "NLP",
                "TAO",
                "Riva",
                "Intent and Slot Classification",
                "TAO Toolkit",
                "BERT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 443298808,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "intentslotclassification_weather_english_bert",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:43:36.984Z"
},{
    "application": "Question Answering",
    "createdDate": "2021-08-18T20:04:58.595Z",
    "description": "Question Answering Bert Large uncased model for extractive question answering on any provided content.",
    "displayName": "Question Answering SQUAD2.0 Bert - Large",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "QA",
                "NLP",
                "SQUAD2.0",
                "TAO",
                "BERT Large",
                "Riva",
                "Question Answering",
                "TAO Toolkit"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 438459496,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "questionanswering_squad_english_bertlarge",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:44:05.680Z"
},{
    "application": "Question Answering",
    "createdDate": "2021-08-18T20:04:58.627Z",
    "description": "Question Answering Bert Base uncased model for extractive question answering on any provided content.",
    "displayName": "Question Answering SQUAD2.0 Bert",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "QA",
                "NLP",
                "SQUAD2.0",
                "TAO",
                "Riva",
                "Question Answering",
                "TAO Toolkit",
                "BERT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 438459496,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "questionanswering_squad_english_bert",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:44:00.138Z"
},{
    "application": "Speech to Text",
    "createdDate": "2021-08-18T20:04:59.381Z",
    "description": "Speech to Text Quartznet model for English.",
    "displayName": "Speech to Text English QuartzNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "ASR",
                "English",
                "STT",
                "Quartznet",
                "TAO",
                "Riva",
                "TAO Toolkit"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.2",
    "latestVersionSizeInBytes": 70904250,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "speechtotext_english_quartznet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-03-26T03:24:08.235Z"
},{
    "application": "Question Answering",
    "createdDate": "2021-08-18T20:05:00.928Z",
    "description": "Question Answering Megatron uncased model for extractive question answering on any provided content.",
    "displayName": "Question Answering SQUAD2.0 Megatron",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "QA",
                "NLP",
                "SQUAD2.0",
                "TAO",
                "Riva",
                "Question Answering",
                "TAO Toolkit",
                "Megatron"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 1337603607,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "questionanswering_squad_english_megatron",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:43:28.489Z"
},{
    "application": "Punctuation and Capitalization",
    "createdDate": "2021-08-18T20:05:02.000Z",
    "description": "For each word in the input text, the model: 1) predicts a punctuation mark that should follow the word (if any), the model supports commas, periods and question marks) and 2) predicts if the word should be capitalized or not.",
    "displayName": "Punctuation and Capitalization Bert",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "Punctuation",
                "NLP",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Capitalization",
                "BERT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 438472343,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "TLT",
    "name": "punctuationcapitalization_english_bert",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-03-26T03:24:29.890Z"
},{
    "application": "OTHER",
    "createdDate": "2021-08-19T02:21:06.246Z",
    "description": "Detect body pose from an image.",
    "displayName": "BodyPoseNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Retail",
                "CV",
                "Metropolis",
                "TAO",
                "Body pose estimation",
                "TAO Toolkit",
                "Healthcare",
                "TLT",
                "Robotics"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0.1",
    "latestVersionSizeInBytes": 67193379,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "bodyposenet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-01-19T22:51:03.295Z"
},{
    "application": "Gesture Classification",
    "createdDate": "2021-08-19T02:21:06.246Z",
    "description": "Classify gestures from hand crop images.",
    "displayName": "GestureNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Retail",
                "CV",
                "Metropolis",
                "TAO",
                "Gesture recognition",
                "TAO Toolkit",
                "Healthcare",
                "Computer Vision",
                "Robotics"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v2.0.1",
    "latestVersionSizeInBytes": 46124617,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "gesturenet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-01-13T18:57:38.868Z"
},{
    "application": "Emotion Classification",
    "createdDate": "2021-08-19T02:21:06.369Z",
    "description": "Network to classify emotions from face.",
    "displayName": "EmotionNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Retail",
                "CV",
                "Metropolis",
                "TAO",
                "TAO Toolkit",
                "Healthcare",
                "Computer Vision",
                "Emotion Recognition"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v1.0",
    "latestVersionSizeInBytes": 4588024,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "emotionnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-01-13T18:59:41.572Z"
},{
    "application": "Object Detection",
    "createdDate": "2021-08-19T02:21:06.369Z",
    "description": "Detect faces from an image.",
    "displayName": "FaceDetect",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Object Detection",
                "Smart City",
                "TAO",
                "Public Safety",
                "Smart Infrastructure",
                "Retail",
                "CV",
                "Metropolis",
                "TAO Toolkit",
                "DetectNet_v2",
                "Healthcare",
                "Computer Vision"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "pruned_quantized_v2.0.1",
    "latestVersionSizeInBytes": 5775090,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "facenet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-01-13T18:58:31.416Z"
},{
    "application": "Fiducial Landmarks",
    "createdDate": "2021-08-19T02:21:06.371Z",
    "description": "Detect fiducial keypoints from an image of a face.",
    "displayName": "Facial Landmarks Estimation",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Retail",
                "CV",
                "Metropolis",
                "TAO",
                "Facial landmark estimation",
                "TAO Toolkit",
                "Computer Vision",
                "Robotics"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v3.0",
    "latestVersionSizeInBytes": 2351014,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "fpenet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2021-11-24T00:50:30.216Z"
},{
    "application": "Riva",
    "createdDate": "2021-08-20T03:26:03.060Z",
    "description": "Base English n-gram LM trained on LibriSpeech, Switchboard and Fisher",
    "displayName": "Riva ASR English LM",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Finetuning"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 6244940310,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechtotext_english_lm",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2021-08-26T15:43:56.947Z"
},{
    "application": "Gaze Detection",
    "createdDate": "2021-08-20T03:53:17.042Z",
    "description": "Detect a persons eye gaze point of regard and gaze vector.",
    "displayName": "Gaze Estimation",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Retail",
                "CV",
                "Metropolis",
                "TAO",
                "TAO Toolkit",
                "Computer Vision",
                "Eye gaze estimation",
                "Robotics"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v1.0",
    "latestVersionSizeInBytes": 18282352,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "gazenet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-01-13T18:59:31.540Z"
},{
    "application": "HeartRateNet Estimation",
    "createdDate": "2021-08-20T20:50:01.480Z",
    "description": "Estimate heart-rate non-invasively from RGB facial videos.",
    "displayName": "HeartRateNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Heart Rate estimation",
                "CV",
                "Metropolis",
                "TAO",
                "TAO Toolkit",
                "Healthcare",
                "Computer Vision"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v2.0",
    "latestVersionSizeInBytes": 588677,
    "logo": "https://raw.githubusercontent.com/kbojo/images/master/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "heartratenet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2021-11-24T00:51:36.328Z"
},{
    "application": "OTHER",
    "createdDate": "2021-08-24T21:13:00.968Z",
    "description": "Semantic segmentation of persons in an image.",
    "displayName": "PeopleSemSegnet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "Smart City",
                "other",
                "TAO",
                "Transfer Learning",
                "FP32",
                "recipe",
                "industry",
                "HDF5",
                "Smart Infrastructure",
                "FP16",
                "Retail",
                "transfer-learning-toolkit",
                "TLT",
                "DeepStream",
                "deep-learning",
                "smart-cities",
                "AI",
                "INT8",
                "technology",
                "image-segmentation",
                "computer-vision",
                "CV",
                "Metropolis",
                "application",
                "TAO Toolkit"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_vanilla_unet_v2.0",
    "latestVersionSizeInBytes": 124469186,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "peoplesemsegnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-06-16T22:07:23.046Z"
},{
    "application": "Text to Speech",
    "createdDate": "2021-08-25T15:09:44.822Z",
    "description": "Mel-Spectrogram prediction conditioned on input text with LJSpeech voice.",
    "displayName": "Speech Synthesis English FastPitch",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "Text to Speech",
                "English",
                "TTS",
                "TAO",
                "Riva",
                "Fastpitch",
                "TAO Toolkit"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.1",
    "latestVersionSizeInBytes": 82356302,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechsynthesis_english_fastpitch",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:45:50.037Z"
},{
    "application": "Text to Speech",
    "createdDate": "2021-08-25T15:09:44.822Z",
    "description": "GAN-based waveform generator from mel-spectrograms.",
    "displayName": "Speech Synthesis HiFi-GAN",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "Text to Speech",
                "TTS",
                "TAO",
                "Riva",
                "HiFiGAN",
                "TAO Toolkit"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 51892640,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechsynthesis_hifigan",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:45:56.910Z"
},{
    "application": "Text to Speech",
    "createdDate": "2021-08-25T15:09:45.349Z",
    "description": "Mel-Spectrogram prediction conditioned on input text with LJSpeech voice.",
    "displayName": "Speech Synthesis English Tacotron2",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "Text to Speech",
                "English",
                "TTS",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Tacotron2"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 112824320,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechsynthesis_english_tacotron2",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:45:53.780Z"
},{
    "application": "Text to Speech",
    "createdDate": "2021-08-25T15:59:16.226Z",
    "description": "Universal waveform generator from mel-spectrograms.",
    "displayName": "Speech Synthesis Waveglow",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "Text to Speech",
                "TTS",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Waveglow"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 342214978,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechsynthesis_waveglow",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:44:11.922Z"
},{
    "application": "Action Recognition",
    "createdDate": "2021-10-22T18:24:05.069Z",
    "description": "5 class action recognition network to recognize what people do in an image.",
    "displayName": "Action Recognition Net",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "Transfer Learning",
                "TAO",
                "CV",
                "Metropolis",
                "AI",
                "TAO Toolkit",
                "Smart Infrastructure",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v2.0",
    "latestVersionSizeInBytes": 310814833,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "actionrecognitionnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-17T00:31:18.396Z"
},{
    "application": "Object Detection",
    "createdDate": "2021-11-23T07:36:59.791Z",
    "description": "Pretrained weights to facilitate transfer learning using TAO Toolkit.",
    "displayName": "TAO Pretrained EfficientDet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "CV",
                "Metropolis",
                "TAO",
                "Transfer Learning",
                "TAO Toolkit",
                "AI",
                "Smart Infrastructure",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        }
    ],
    "latestVersionIdStr": "efficientnet_b2",
    "latestVersionSizeInBytes": 64864720,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "pretrained_efficientdet",
    "orgName": "nvidia",
    "precision": "FP32",
    "teamName": "tao",
    "updatedDate": "2021-11-24T20:13:37.840Z"
},{
    "application": "Pose Estimation",
    "createdDate": "2021-12-06T00:47:27.212Z",
    "description": "3D human pose estimation network to predict 34 keypoints in 3D of a person in an image.",
    "displayName": "BodyPose3DNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "CV",
                "Metropolis",
                "Transfer Learning",
                "TAO Toolkit",
                "Smart Cities",
                "AI",
                "Smart Infrastructure",
                "Computer Vision",
                "Transfer Learning Toolkit",
                "Deep Learning"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_performance_v1.0",
    "latestVersionSizeInBytes": 40360415,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "bodypose3dnet",
    "orgName": "nvidia",
    "precision": "FP16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2021-12-09T20:57:24.893Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-01-06T17:12:08.183Z",
    "description": "English Citrinet ASR model trained on ASR set 3.0, no-weight-decay",
    "displayName": "RIVA Citrinet ASR English",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "English",
                "STT",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Citrinet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v3.0",
    "latestVersionSizeInBytes": 566722587,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "riva",
    "name": "speechtotext_en_us_citrinet",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-02T07:48:44.918Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-01-06T17:15:39.660Z",
    "description": "Spanish Citrinet ASR model trained on ASR set 2.0",
    "displayName": "RIVA Citrinet ASR Spanish",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "STT",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Spanish",
                "Citrinet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "AMP"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v2.0",
    "latestVersionSizeInBytes": 566726427,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "riva",
    "name": "speechtotext_es_us_citrinet",
    "orgName": "nvidia",
    "precision": "AMP",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-02T07:34:20.885Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-01-06T17:17:07.715Z",
    "description": "German Citrinet ASR model trained on ASR set 2.0",
    "displayName": "RIVA Citrinet ASR German",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "STT",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "German",
                "Citrinet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v2.0",
    "latestVersionSizeInBytes": 566726189,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "riva",
    "name": "speechtotext_de_de_citrinet",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-02T08:02:12.892Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-01-06T17:20:04.171Z",
    "description": "Russian Citrinet ASR model trained on ASR set 1.0",
    "displayName": "RIVA Citrinet ASR Russian",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "STT",
                "Russian",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Citrinet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v1.0",
    "latestVersionSizeInBytes": 566725641,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "riva",
    "name": "speechtotext_ru_ru_citrinet",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-02T08:01:23.514Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-01-06T18:04:05.681Z",
    "description": "English Conformer ASR model trained on ASR set 3.0",
    "displayName": "RIVA Conformer ASR English - ASR set 3.0",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "English",
                "STT",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Conformer"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v3.0",
    "latestVersionSizeInBytes": 488589690,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "riva",
    "name": "speechtotext_en_us_conformer",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-02T08:00:35.255Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-01-06T18:06:44.126Z",
    "description": "English Quartznet ASR model trained on ASR set 1.2",
    "displayName": "RIVA Quartznet ASR English",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "English",
                "STT",
                "Quartznet",
                "TAO",
                "Riva",
                "TAO Toolkit"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.2",
    "latestVersionSizeInBytes": 70904250,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "riva",
    "name": "speechtotext_en_us_quartznet",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:45:00.526Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-01-06T18:08:01.771Z",
    "description": "English ASR model trained on ASR Set 1.2, Noise Robust",
    "displayName": "RIVA Jasper ASR English",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "English",
                "Jasper",
                "STT",
                "TAO",
                "Riva",
                "TAO Toolkit"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.2",
    "latestVersionSizeInBytes": 1234711099,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "riva",
    "name": "speechtotext_en_us_jasper",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:45:14.736Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-01-06T18:12:16.343Z",
    "description": "Base English n-gram LM trained on LibriSpeech, Switchboard and Fisher",
    "displayName": "Riva ASR English LM",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "English",
                "LM",
                "TAO",
                "Riva",
                "TAO Toolkit"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "n/a"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.1",
    "latestVersionSizeInBytes": 6261530037,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "ARPA",
    "name": "speechtotext_en_us_lm",
    "orgName": "nvidia",
    "precision": "n/a",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:45:04.359Z"
},{
    "application": "NVIDIA Riva EA",
    "builtBy": "aiapps",
    "createdDate": "2022-03-09T01:27:18.447Z",
    "description": "Contains files used in rmir creation",
    "displayName": "Riva TTS English US Auxiliary Files",
    "framework": "Riva",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "Riva",
                "Conversational AI",
                "Speech Synthesis"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Riva"
            ]
        },
        {
            "key": "precision",
            "values": [
                "n/a"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.1",
    "latestVersionSizeInBytes": 3721630,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "n/a",
    "name": "speechsynthesis_en_us_auxiliary_files",
    "orgName": "nvidia",
    "precision": "n/a",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-25T14:50:28.672Z"
},{
    "application": "NVIDIA Riva EA",
    "builtBy": "aiapps",
    "createdDate": "2022-03-17T23:00:07.376Z",
    "description": "Base German 4-gram LM",
    "displayName": "Riva ASR German LM",
    "framework": "Riva",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "German",
                "Citrinet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Riva"
            ]
        },
        {
            "key": "precision",
            "values": [
                "n/a"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v2.0",
    "latestVersionSizeInBytes": 1393581375,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "ARPA",
    "name": "speechtotext_de_de_lm",
    "orgName": "nvidia",
    "precision": "n/a",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:44:08.874Z"
},{
    "application": "NVIDIA Riva EA",
    "builtBy": "aiapps",
    "createdDate": "2022-03-17T23:02:17.225Z",
    "description": "Base Russian 4-gram LM",
    "displayName": "Riva ASR Russian LM",
    "framework": "Riva",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "Russian",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Citrinet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Riva"
            ]
        },
        {
            "key": "precision",
            "values": [
                "n/a"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.1",
    "latestVersionSizeInBytes": 2681513539,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "ARPA",
    "name": "speechtotext_ru_ru_lm",
    "orgName": "nvidia",
    "precision": "n/a",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-06-15T10:25:43.966Z"
},{
    "application": "NVIDIA Riva EA",
    "builtBy": "aiapps",
    "createdDate": "2022-03-17T23:05:19.998Z",
    "description": "Base Spanish 4-gram LM",
    "displayName": "Riva ASR Spanish LM",
    "framework": "Riva",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Spanish",
                "Citrinet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Riva"
            ]
        },
        {
            "key": "precision",
            "values": [
                "n/a"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v2.0",
    "latestVersionSizeInBytes": 1067128899,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "ARPA",
    "name": "speechtotext_es_us_lm",
    "orgName": "nvidia",
    "precision": "n/a",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:44:48.578Z"
},{
    "application": "Text to Speech",
    "builtBy": "aiapps",
    "createdDate": "2022-03-22T00:32:05.604Z",
    "description": "Mel-Spectrogram prediction conditioned on input text with English US Male 1 voice.",
    "displayName": "RIVA English Fastpitch Male 1",
    "framework": "NeMo",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "English",
                "TAO",
                "Riva",
                "Fastpitch",
                "Speech Synthesis"
            ]
        },
        {
            "key": "framework",
            "values": [
                "NeMo"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 90673011,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechsynthesis_en_us_fastpitch_male_1",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-19T13:46:11.073Z"
},{
    "application": "Text to Speech",
    "builtBy": "aiapps",
    "createdDate": "2022-03-22T00:32:47.475Z",
    "description": "Mel-Spectrogram prediction conditioned on input text with English US Female 1 voice.",
    "displayName": "RIVA English Fastpitch Female 1",
    "framework": "NeMo",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "English",
                "TAO",
                "Riva",
                "Fastpitch",
                "Speech Synthesis"
            ]
        },
        {
            "key": "framework",
            "values": [
                "NeMo"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 90672542,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechsynthesis_en_us_fastpitch_female_1",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-19T13:46:53.333Z"
},{
    "application": "Text to Speech",
    "builtBy": "aiapps",
    "createdDate": "2022-03-22T00:40:20.202Z",
    "description": "GAN-based waveform generator from mel-spectrograms.",
    "displayName": "RIVA Hifigan Male 1",
    "framework": "NeMo",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "TTS",
                "TAO",
                "Riva",
                "HiFiGAN",
                "Speech Synthesis"
            ]
        },
        {
            "key": "framework",
            "values": [
                "NeMo"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 55755743,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechsynthesis_en_us_hifigan_male_1",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:45:25.847Z"
},{
    "application": "Text to Speech",
    "builtBy": "aiapps",
    "createdDate": "2022-03-22T00:41:40.605Z",
    "description": "GAN-based waveform generator from mel-spectrograms.",
    "displayName": "RIVA Hifigan Female 1",
    "framework": "NeMo",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "TTS",
                "TAO",
                "Riva",
                "HiFiGAN",
                "Speech Synthesis"
            ]
        },
        {
            "key": "framework",
            "values": [
                "NeMo"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 55755675,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechsynthesis_en_us_hifigan_female_1",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:45:29.007Z"
},{
    "application": "Speech to Text",
    "createdDate": "2022-03-23T11:50:58.191Z",
    "description": "German Conformer ASR model trained on ASR set 2.0",
    "displayName": "RIVA Conformer ASR German",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "ASR",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "German",
                "Conformer"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v2.0",
    "latestVersionSizeInBytes": 488591508,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechtotext_de_de_conformer",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-02T07:40:01.043Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-03-29T23:17:55.307Z",
    "description": "Mandarin Citrinet ASR model trained on ASR set 2.0",
    "displayName": "RIVA Citrinet ASR Mandarin",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Mandarin",
                "Citrinet"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v2.1",
    "latestVersionSizeInBytes": 583781459,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "riva",
    "name": "speechtotext_zh_cn_citrinet",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-02T09:51:30.900Z"
},{
    "application": "Punctuation and Capitalization",
    "builtBy": "aiapps",
    "createdDate": "2022-03-31T21:36:59.700Z",
    "description": "For each word in the input text, the model: 1) predicts a punctuation mark that should follow the word (if any), the model supports commas, periods and question marks) and 2) predicts if the word should be capitalized or not.",
    "displayName": "RIVA Punctuation and Capitalization for German",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "TAO",
                "Transfer Learning",
                "Conversational-Ai",
                "Riva",
                "Capitalization",
                "Inference",
                "German",
                "Punctuation",
                "NLP",
                "TAO Toolkit",
                "BERT",
                "Finetuning",
                "Natural-Language-Processing"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v1.0",
    "latestVersionSizeInBytes": 712286911,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "punctuationcapitalization_de_de_bert_base",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-11T15:32:27.081Z"
},{
    "application": "Punctuation and Capitalization",
    "builtBy": "aiapps",
    "createdDate": "2022-03-31T21:37:40.065Z",
    "description": "For each word in the input text, the model: 1) predicts a punctuation mark that should follow the word (if any), the model supports commas, periods and question marks) and 2) predicts if the word should be capitalized or not.",
    "displayName": "RIVA Punctuation and Capitalization for Spanish",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "TAO",
                "Transfer Learning",
                "Conversational-Ai",
                "Riva",
                "Capitalization",
                "Inference",
                "Punctuation",
                "NLP",
                "TAO Toolkit",
                "BERT",
                "Spanish",
                "Finetuning",
                "Natural-Language-Processing"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v1.0",
    "latestVersionSizeInBytes": 712289648,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "punctuationcapitalization_es_us_bert_base",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-11T15:30:34.728Z"
},{
    "application": "Punctuation and Capitalization",
    "builtBy": "aiapps",
    "createdDate": "2022-03-31T21:38:24.169Z",
    "description": "For each word in the input text, the model: 1) predicts a punctuation mark that should follow the word (if any), the model supports commas, periods and question marks) and 2) predicts if the word should be capitalized or not.",
    "displayName": "RIVA Punctuation",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "Punctuation",
                "NLP",
                "TAO",
                "Transfer Learning",
                "Conversational-Ai",
                "Riva",
                "TAO Toolkit",
                "Capitalization",
                "Inference",
                "BERT",
                "Finetuning",
                "Natural-Language-Processing"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 438472343,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "punctuationcapitalization_en_us_bert_base",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:43:53.336Z"
},{
    "application": "NVIDIA Riva",
    "builtBy": "aiapps",
    "createdDate": "2022-04-05T23:21:15.510Z",
    "description": "Base English grammar",
    "displayName": "Riva ASR English Inverse Normalization Grammar",
    "framework": "Riva",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Riva"
            ]
        },
        {
            "key": "precision",
            "values": [
                "n/a"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.1",
    "latestVersionSizeInBytes": 1350491,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "FAR",
    "name": "inverse_normalization_en_us",
    "orgName": "nvidia",
    "precision": "n/a",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-06-16T20:57:00.744Z"
},{
    "application": "NVIDIA Riva",
    "builtBy": "aiapps",
    "createdDate": "2022-04-05T23:21:38.219Z",
    "description": "Base Spanish grammar",
    "displayName": "Riva ASR Spanish Inverse Normalization Grammar",
    "framework": "Riva",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Riva"
            ]
        },
        {
            "key": "precision",
            "values": [
                "n/a"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 1386195,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "FAR",
    "name": "inverse_normalization_es_us",
    "orgName": "nvidia",
    "precision": "n/a",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:44:25.262Z"
},{
    "application": "NVIDIA Riva",
    "builtBy": "aiapps",
    "createdDate": "2022-04-05T23:21:53.250Z",
    "description": "Base German grammar",
    "displayName": "Riva ASR German Inverse Normalization Grammar",
    "framework": "Riva",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Riva"
            ]
        },
        {
            "key": "precision",
            "values": [
                "n/a"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 4128539,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "FAR",
    "name": "inverse_normalization_de_de",
    "orgName": "nvidia",
    "precision": "n/a",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T02:44:33.127Z"
},{
    "application": "NVIDIA Riva",
    "builtBy": "aiapps",
    "createdDate": "2022-04-05T23:22:07.127Z",
    "description": "Base English grammar",
    "displayName": "Riva TTS English Normalization Grammar",
    "framework": "Riva",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Riva"
            ]
        },
        {
            "key": "precision",
            "values": [
                "n/a"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.1",
    "latestVersionSizeInBytes": 2390007,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "FAR",
    "name": "normalization_en_us",
    "orgName": "nvidia",
    "precision": "n/a",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-20T15:16:58.353Z"
},{
    "application": "Speech To Text",
    "builtBy": "aiapps",
    "createdDate": "2022-04-05T23:25:16.549Z",
    "description": "Base Mandarin 4-gram LM",
    "displayName": "Riva ASR Mandarin LM",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "LM",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Automatic_speech_recognition",
                "Mandarin",
                "Conversational_ai"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "NA"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v2.1",
    "latestVersionSizeInBytes": 8461503449,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "arpa",
    "name": "speechtotext_zh_cn_lm",
    "orgName": "nvidia",
    "precision": "NA",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-26T08:11:10.464Z"
},{
    "application": "Speech To Text",
    "builtBy": "aiapps",
    "createdDate": "2022-04-08T04:37:39.406Z",
    "description": "English Citrinet-256 ASR model trained on ASR set 2.0, no-weight-decay",
    "displayName": "RIVA Citrinet 256 ASR English",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Automatic_speech_recognition",
                "Conversational_ai"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v2.0",
    "latestVersionSizeInBytes": 41140788,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "speechtotext_en_us_citrinet256",
    "orgName": "nvidia",
    "precision": "FP16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T04:49:11.919Z"
},{
    "application": "Joint Intent And Slot Classification",
    "builtBy": "aiapps",
    "createdDate": "2022-04-08T04:38:25.037Z",
    "description": "Intent and Slot classification of the queries for the misty bot with DistilBert model trained on weather, smalltalk and POI (places of interest) data.",
    "displayName": "Joint Intent and Slot Classification DistilBert",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "NLP",
                "TAO",
                "Riva",
                "Natural Language Processing",
                "Intent and slot classification",
                "TAO Toolkit",
                "BERT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 266351975,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "intentslotclassification_misty_english_distilbert",
    "orgName": "nvidia",
    "precision": "FP16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-04-08T04:50:25.861Z"
},{
    "application": "Speech to Text",
    "builtBy": "aiapps",
    "createdDate": "2022-04-20T17:44:38.494Z",
    "description": "Spanish Conformer ASR model trained on ASR set 2.0",
    "displayName": "RIVA Conformer ASR Spanish",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "builtBy",
            "values": [
                "aiapps"
            ]
        },
        {
            "key": "general",
            "values": [
                "ASR",
                "TAO",
                "Riva",
                "TAO Toolkit",
                "Spanish",
                "Conformer"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "fp16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "trainable_v2.1",
    "latestVersionSizeInBytes": 486998134,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "riva",
    "name": "speechtotext_es_us_conformer",
    "orgName": "nvidia",
    "precision": "fp16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-02T09:51:30.819Z"
},{
    "application": "Object Detection",
    "createdDate": "2022-05-12T22:31:10.383Z",
    "description": "Model to detect one or more objects from a LIDAR point cloud file and return 3D bounding boxes.",
    "displayName": "PointPillarNet",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "CV",
                "Metropolis",
                "TAO",
                "Transfer Learning",
                "TAO Toolkit",
                "AI",
                "Smart Infrastructure",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 5572394,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "pointpillarnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "teamName": "tao",
    "updatedDate": "2022-05-12T22:31:34.034Z"
},{
    "application": "Pose Classification",
    "createdDate": "2022-05-12T22:31:11.382Z",
    "description": "Pose classification network to classify poses of people from their skeletons.",
    "displayName": "Pose Classification",
    "framework": "Transfer Learning Toolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "DeepStream",
                "Smart City",
                "CV",
                "Metropolis",
                "TAO",
                "Transfer Learning",
                "TAO Toolkit",
                "AI",
                "Smart Infrastructure",
                "TLT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "Transfer Learning Toolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP32"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 12730328,
    "logo": "https://dz112fgwz7ogh.cloudfront.net/logos/Nvidia-Centric-TAO.png",
    "modelFormat": "TLT",
    "name": "poseclassificationnet",
    "orgName": "nvidia",
    "precision": "FP32",
    "teamName": "tao",
    "updatedDate": "2022-05-12T22:31:26.218Z"
},{
    "application": "NLP",
    "createdDate": "2022-05-12T22:50:14.594Z",
    "description": "Intent and Slot classification of the queries for the misty bot with BERT model trained on weather, smalltalk and POI (places of interest) data.",
    "displayName": "Joint Intent and Slot Classification Misty Bert",
    "framework": "TransferLearningToolkit",
    "isPublic": true,
    "labels": [
        {
            "key": "general",
            "values": [
                "NLP",
                "TAO",
                "Riva",
                "Intent and Slot Classification",
                "Natural Language Processing",
                "TAO Toolkit",
                "BERT"
            ]
        },
        {
            "key": "framework",
            "values": [
                "TransferLearningToolkit"
            ]
        },
        {
            "key": "precision",
            "values": [
                "FP16"
            ]
        },
        {
            "key": "publisher",
            "values": [
                "NVIDIA"
            ]
        }
    ],
    "latestVersionIdStr": "deployable_v1.0",
    "latestVersionSizeInBytes": 438931389,
    "logo": "https://github.com/kbojo/images/raw/master/Nvidia-Centric-TAO-RIVA.png",
    "modelFormat": "RIVA",
    "name": "intentslotclassification_misty_english_bert",
    "orgName": "nvidia",
    "precision": "FP16",
    "publisher": "NVIDIA",
    "teamName": "tao",
    "updatedDate": "2022-05-12T22:52:30.321Z"
}]
dashcamnet
# 2.3
!ngc registry model download-version nvidia/tao/dashcamnet:pruned_v1.0 --dest $NGC_DIR \
2>&1| tee my_assessment/answer_2.txt
{
    "download_end": "2022-07-10 10:32:59.423741",
    "download_start": "2022-07-10 10:32:56.419345",
    "download_time": "3s",
    "files_downloaded": 3,
    "local_path": "/dli/task/ngc_assets/dashcamnet_vpruned_v1.0",
    "size_downloaded": "6.65 MB",
    "status": "Completed",
    "transfer_id": "dashcamnet_vpruned_v1.0"
}
### Step 3: Edit the Inference Configuration File ###
The next step is to modify the Gst-nvinfer configuration file that will be used to configure the AI inference plugin. You can create a new text file for this purpose manually and start from scratch or use the [template provided](spec_files/pgie_config_dashcamnet.txt). You can also refer to sample applications and configuration files [here](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps). When creating the configuration file, below are the fields to pay attention to: 
​
Following properties are used when using TAO Toolkit models downloaded from NGC: 
* `tlt-encoded-model` - Pathname of the TAO Toolkit encoded model
* `tlt-model-key` - Model load key for the TAO Toolkit encoded model
* `labelfile-path` - Pathname of a text file containing the labels for the model
* `int8-calib-file` - Pathname of the INT8 calibration file for dynamic range adjustment with an FP32 model (only in INT8)
* `uff-input-blob-name` - Name of the input blob in the UFF file
* `output-blob-names` - Array of output layer names
* `input-dims` - Dimensions of the model as [channel; height; width; input-order] if input-order=0 i.e. NCHW
* `net-scale-factor` - Pixel normalization factor _(default=1)_
​
Recommended properties: 
* `batch-size` - Number of frames to be inferred together in a batch _(default=1)_
​
Mandatory properties for detectors: 
* `num-detected-classes` - Number of classes detected by the network
​
Optional properties for detectors: 
* `cluster-mode` - Clustering algorithm to use _(default=0 i.e. Group Rectangles)_
* `interval` - Number of consecutive batches to be skipped for inference _(primary mode only | default=0)_
​
Other optional properties: 
* `network-mode` - Data format to be used for inference _(0=FP32, 1=INT8, 2=FP16 mode | default=0 i.e. FP32)_
* `process-mode` - Mode _(primary or secondary)_ in which the plugin is to operate on _(default=1 i.e. primary)_
* `model-color-format` - Color format required by the model _(default=0 i.e. RGB)_
* `gie-unique-id` - Unique ID to be assigned to the GIE to enable the application and other elements to identify detected bounding boxes and labels _(default=0)_
* `model-engine-file` - Pathname of the serialized model engine file
* `gpu-id` - Device ID of GPU to use for pre-processing/inference _(dGPU only)_
​
**Note**: The values in the config file are overridden by values set through GObject properties. Another important thing to remember is that the properties recommended are specific to a primary detector, you will need to work on other properties for secondary and/or classifier. You can find most of the information needed on the [model card](https://catalog.ngc.nvidia.com/orgs/nvidia/models/tlt_dashcamnet): 
​
<p><img src='images/model_card.png' width=720></p>
**Instructions**: 
<br>
3.1. Open and review the [configuration file](spec_files/pgie_config_dashcamnet.txt). <br>
3.2. Update the `<FIXME>`s _only_ in the configuration file with the correct values and **save changes**. Afterwards, make sure in the cell the correct path of the configuration file is referenced and execute the cell to mark your answer. _You can execute this cell multiple times until satisfactory_. <br>
# 3.2
os.environ['SPEC_FILE']='/dli/task/spec_files/pgie_config_dashcamnet.txt'
​
# DO NOT CHANGE BELOW
!cat $SPEC_FILE
!cp -v $SPEC_FILE my_assessment/answer_3.txt
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
tlt-model-key=tlt_encode
tlt-encoded-model=/dli/task/ngc_assets/dashcamnet_vpruned_v1.0/resnet18_dashcamnet_pruned.etlt
labelfile-path=/dli/task/ngc_assets/dashcamnet_vpruned_v1.0/labels.txt
int8-calib-file=/dli/task/ngc_assets/dashcamnet_vpruned_v1.0/dashcamnet_int8.txt
input-dims=3;720;1280;0
uff-input-blob-name=input_1
batch-size=1
process-mode=1
model-color-format=0
# 0=FP32, 1=INT8, 2=FP16 mode
network-mode=2
num-detected-classes=4
interval=0
gie-unique-id=1
output-blob-names=output_bbox/BiasAdd;output_cov/Sigmoid
cluster-mode=0

[class-attrs-all]
pre-cluster-threshold=0.1
## Set eps=0.7 and minBoxes for cluster-mode=1(DBSCAN)
eps=0.7
minBoxes=1'/dli/task/spec_files/pgie_config_dashcamnet.txt' -> 'my_assessment/answer_3.txt'
### Step 4: Build and Run DeepStream Pipeline ###
Next, it's time to build the pipeline. We're putting the pipeline creation and initiation procedure inside a function so it an be called easily. We also need to implement the probe callback function prior to running the pipeline. We've provided you a functional architecture and framework for this application to follow. Below is the architecture for this pipeline. 
​
<p><img src='images/assessment_pipeline.png' width=1080></p>
​
Our logic for determining if a vehicle is tailgating will be based on the coordinates of detected objects' bounding boxes shown below: 
​
<p><img src='images/tailgate_metrics.png' width=720></p>
​
While we attached the probe to the _nvdsosd_ plugin, the only requirement is that it has to be after the _nvinfer_ plugin so it contains the AI-infered metadata. Recall that we need to program the probe [callback function](https://en.wikipedia.org/wiki/Callback_(computer_programming)) to provide us a signal when tailgating is potentially occuring. The probe callback function generally follows a boilerplate, to help iterate through the batches, frames, and objects. For more information on how to implement a callback function, please refer to the [GStreamer Probe documentation](https://gstreamer.freedesktop.org/documentation/additional/design/probes.html). 
​
<p><img src='images/probe_boiler_plate.png' width=720></p>
​
We want to generate a list that will contain 0s and 1s for each frame to represent if it exhibits tailgating. Therefore there should be as many numbers as there are number of frames in the end. There should _not_ be one number associated with each object detected as it will lead to more than one number associated with each frame. Below is a sample output: 
​
<p><img src='images/sample_log.png' width=720></p>
**Instructions**: 
<br>
4.1. Review the pipeline architecture. <br>
4.2. Modify the `<FIXME>` _only_ in the cell with the correct code and execute the cell to define the function that will build and run the pipeline. <br>
4.3. Modify the `<FIXME>` _only_ in the cell with the correct code and execute the cell to define the probe callback function. <br>
4.4. Execute the cell to run the pipeline. <br>
# 4.2
#Import necessary libraries
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from common.bus_call import bus_call
import pyds
​
def run(input_file_path):
    global inference_output
    inference_output=[]
    Gst.init(None)
​
    # Create element that will form a pipeline
    print("Creating Pipeline")
    pipeline=Gst.Pipeline()
    
    source=Gst.ElementFactory.make("filesrc", "file-source")
    source.set_property('location', input_file_path)
    h264parser=Gst.ElementFactory.make("h264parse", "h264-parser")
    decoder=Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    
    streammux=Gst.ElementFactory.make("nvstreammux", "Stream-muxer")    
    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    
    pgie=Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', os.environ['SPEC_FILE'])
    
    nvvidconv1=Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd=Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvvidconv2=Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    capsfilter=Gst.ElementFactory.make("capsfilter", "capsfilter")
    caps=Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)
    
    encoder=Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    encoder.set_property("bitrate", 2000000)
    
    sink=Gst.ElementFactory.make("filesink", 'filesink')
    sink.set_property('location', 'output.mpeg4')
    sink.set_property("sync", 1)
    
    # Add the elements to the pipeline
    print("Adding elements to Pipeline")
    pipeline.add(source)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(sink)
​
    # Link the elements together
    print("Linking elements in the Pipeline")
    source.link(h264parser)
    h264parser.link(decoder)
    decoder.get_static_pad('src').link(streammux.get_request_pad("sink_0"))
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(nvosd)
    nvosd.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(sink)
    
    # Attach probe to OSD sink pad
    osdsinkpad=nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
​
    # Create an event loop and feed gstreamer bus mesages to it
    loop=GLib.MainLoop()
    bus=pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    
    # Start play back and listen to events
    print("Starting pipeline")
    
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    
    pipeline.set_state(Gst.State.NULL)
    return inference_output
36
# 4.3
# Define the Probe Function
def osd_sink_pad_buffer_probe(pad, info, u_data):
    gst_buffer=info.get_buffer()
​
    # Retrieve batch metadata from the gst_buffer
    batch_meta=pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame=batch_meta.frame_meta_list
    while l_frame is not None:
        
        # Initially set the tailgate indicator to False for each frame
        tailgate=False
        try:
            frame_meta=pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number=frame_meta.frame_num
        l_obj=frame_meta.obj_meta_list
        
        # Iterate through each object to check its dimension
        while l_obj is not None:
            try:
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                
                # If the object meet the criteria then set tailgate indicator to True
                obj_bottom=obj_meta.rect_params.top+obj_meta.rect_params.height
                if (obj_meta.rect_params.width > FRAME_WIDTH*.3) & (obj_bottom > FRAME_HEIGHT*.9): 
                    tailgate=True
                    
            except StopIteration:
                break
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
                
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Tailgate={}".format(frame_number, tailgate)
​
        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
​
        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 36
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
​
        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        print(f'Analyzing frame {frame_number}', end='\r')
        inference_output.append(str(int(tailgate)))
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK
# 4.4
tailgate_log=run(input_file_path='/dli/task/data/assessment_stream.h264')
​
# DO NOT CHANGE BELOW
with open('/dli/task/my_assessment/answer_4.txt', 'w') as f: 
    f.write('\n'.join(tailgate_log))
Creating Pipeline
Adding elements to Pipeline
Linking elements in the Pipeline
Starting pipeline
Frame Number=0 Tailgate=False
Frame Number=1 Tailgate=False
Frame Number=2 Tailgate=False
Frame Number=3 Tailgate=False
Frame Number=4 Tailgate=False
Frame Number=5 Tailgate=False
Frame Number=6 Tailgate=False
Frame Number=7 Tailgate=False
Frame Number=8 Tailgate=False
Frame Number=9 Tailgate=False
Frame Number=10 Tailgate=False
Frame Number=11 Tailgate=False
Frame Number=12 Tailgate=False
Frame Number=13 Tailgate=False
Frame Number=14 Tailgate=False
Frame Number=15 Tailgate=False
Frame Number=16 Tailgate=False
Frame Number=17 Tailgate=False
Frame Number=18 Tailgate=False
Frame Number=19 Tailgate=False
Frame Number=20 Tailgate=False
Frame Number=21 Tailgate=False
Frame Number=22 Tailgate=False
Frame Number=23 Tailgate=False
Frame Number=24 Tailgate=False
Frame Number=25 Tailgate=False
Frame Number=26 Tailgate=False
Frame Number=27 Tailgate=False
Frame Number=28 Tailgate=False
Frame Number=29 Tailgate=False
Frame Number=30 Tailgate=False
Frame Number=31 Tailgate=False
Frame Number=32 Tailgate=False
Frame Number=33 Tailgate=False
Frame Number=34 Tailgate=False
Frame Number=35 Tailgate=False
Frame Number=36 Tailgate=False
Frame Number=37 Tailgate=False
Frame Number=38 Tailgate=False
Frame Number=39 Tailgate=False
Frame Number=40 Tailgate=False
Frame Number=41 Tailgate=False
Frame Number=42 Tailgate=False
Frame Number=43 Tailgate=False
Frame Number=44 Tailgate=False
Frame Number=45 Tailgate=False
Frame Number=46 Tailgate=False
Frame Number=47 Tailgate=False
Frame Number=48 Tailgate=False
Frame Number=49 Tailgate=False
Frame Number=50 Tailgate=False
Frame Number=51 Tailgate=False
Frame Number=52 Tailgate=False
Frame Number=53 Tailgate=False
Frame Number=54 Tailgate=False
Frame Number=55 Tailgate=False
Frame Number=56 Tailgate=False
Frame Number=57 Tailgate=False
Frame Number=58 Tailgate=False
Frame Number=59 Tailgate=False
Frame Number=60 Tailgate=False
Frame Number=61 Tailgate=False
Frame Number=62 Tailgate=False
Frame Number=63 Tailgate=False
Frame Number=64 Tailgate=False
Frame Number=65 Tailgate=False
Frame Number=66 Tailgate=False
Frame Number=67 Tailgate=False
Frame Number=68 Tailgate=False
Frame Number=69 Tailgate=False
Frame Number=70 Tailgate=False
Frame Number=71 Tailgate=False
Frame Number=72 Tailgate=False
Frame Number=73 Tailgate=False
Frame Number=74 Tailgate=False
Frame Number=75 Tailgate=False
Frame Number=76 Tailgate=False
Frame Number=77 Tailgate=False
Frame Number=78 Tailgate=False
Frame Number=79 Tailgate=False
Frame Number=80 Tailgate=False
Frame Number=81 Tailgate=False
Frame Number=82 Tailgate=False
Frame Number=83 Tailgate=False
Frame Number=84 Tailgate=False
Frame Number=85 Tailgate=False
Frame Number=86 Tailgate=False
Frame Number=87 Tailgate=False
Frame Number=88 Tailgate=False
Frame Number=89 Tailgate=False
Frame Number=90 Tailgate=False
Frame Number=91 Tailgate=False
Frame Number=92 Tailgate=False
Frame Number=93 Tailgate=False
Frame Number=94 Tailgate=False
Frame Number=95 Tailgate=False
Frame Number=96 Tailgate=False
Frame Number=97 Tailgate=False
Frame Number=98 Tailgate=False
Frame Number=99 Tailgate=False
Frame Number=100 Tailgate=False
Frame Number=101 Tailgate=False
Frame Number=102 Tailgate=False
Frame Number=103 Tailgate=False
Frame Number=104 Tailgate=False
Frame Number=105 Tailgate=False
Frame Number=106 Tailgate=False
Frame Number=107 Tailgate=False
Frame Number=108 Tailgate=False
Frame Number=109 Tailgate=False
Frame Number=110 Tailgate=False
Frame Number=111 Tailgate=False
Frame Number=112 Tailgate=False
Frame Number=113 Tailgate=False
Frame Number=114 Tailgate=False
Frame Number=115 Tailgate=False
Frame Number=116 Tailgate=False
Frame Number=117 Tailgate=False
Frame Number=118 Tailgate=False
Frame Number=119 Tailgate=False
Frame Number=120 Tailgate=False
Frame Number=121 Tailgate=False
Frame Number=122 Tailgate=False
Frame Number=123 Tailgate=False
Frame Number=124 Tailgate=False
Frame Number=125 Tailgate=False
Frame Number=126 Tailgate=False
Frame Number=127 Tailgate=False
Frame Number=128 Tailgate=False
Frame Number=129 Tailgate=False
Frame Number=130 Tailgate=False
Frame Number=131 Tailgate=False
Frame Number=132 Tailgate=False
Frame Number=133 Tailgate=False
Frame Number=134 Tailgate=False
Frame Number=135 Tailgate=False
Frame Number=136 Tailgate=False
Frame Number=137 Tailgate=False
Frame Number=138 Tailgate=False
Frame Number=139 Tailgate=False
Frame Number=140 Tailgate=False
Frame Number=141 Tailgate=False
Frame Number=142 Tailgate=False
Frame Number=143 Tailgate=False
Frame Number=144 Tailgate=False
Frame Number=145 Tailgate=False
Frame Number=146 Tailgate=False
Frame Number=147 Tailgate=False
Frame Number=148 Tailgate=False
Frame Number=149 Tailgate=False
Frame Number=150 Tailgate=False
Frame Number=151 Tailgate=False
Frame Number=152 Tailgate=False
Frame Number=153 Tailgate=False
Frame Number=154 Tailgate=False
Frame Number=155 Tailgate=False
Frame Number=156 Tailgate=False
Frame Number=157 Tailgate=False
Frame Number=158 Tailgate=False
Frame Number=159 Tailgate=False
Frame Number=160 Tailgate=False
Frame Number=161 Tailgate=False
Frame Number=162 Tailgate=False
Frame Number=163 Tailgate=False
Frame Number=164 Tailgate=False
Frame Number=165 Tailgate=False
Frame Number=166 Tailgate=False
Frame Number=167 Tailgate=False
Frame Number=168 Tailgate=False
Frame Number=169 Tailgate=False
Frame Number=170 Tailgate=False
Frame Number=171 Tailgate=False
Frame Number=172 Tailgate=False
Frame Number=173 Tailgate=False
Frame Number=174 Tailgate=False
Frame Number=175 Tailgate=False
Frame Number=176 Tailgate=False
Frame Number=177 Tailgate=False
Frame Number=178 Tailgate=False
Frame Number=179 Tailgate=False
Frame Number=180 Tailgate=False
Frame Number=181 Tailgate=False
Frame Number=182 Tailgate=False
Frame Number=183 Tailgate=False
Frame Number=184 Tailgate=False
Frame Number=185 Tailgate=False
Frame Number=186 Tailgate=False
Frame Number=187 Tailgate=False
Frame Number=188 Tailgate=False
Frame Number=189 Tailgate=False
Frame Number=190 Tailgate=False
Frame Number=191 Tailgate=False
Frame Number=192 Tailgate=False
Frame Number=193 Tailgate=False
Frame Number=194 Tailgate=False
Frame Number=195 Tailgate=False
Frame Number=196 Tailgate=False
Frame Number=197 Tailgate=False
Frame Number=198 Tailgate=False
Frame Number=199 Tailgate=False
Frame Number=200 Tailgate=False
Frame Number=201 Tailgate=False
Frame Number=202 Tailgate=False
Frame Number=203 Tailgate=False
Frame Number=204 Tailgate=False
Frame Number=205 Tailgate=False
Frame Number=206 Tailgate=False
Frame Number=207 Tailgate=False
Frame Number=208 Tailgate=False
Frame Number=209 Tailgate=False
Frame Number=210 Tailgate=False
Frame Number=211 Tailgate=False
Frame Number=212 Tailgate=False
Frame Number=213 Tailgate=False
Frame Number=214 Tailgate=False
Frame Number=215 Tailgate=False
Frame Number=216 Tailgate=False
Frame Number=217 Tailgate=False
Frame Number=218 Tailgate=False
Frame Number=219 Tailgate=False
Frame Number=220 Tailgate=False
Frame Number=221 Tailgate=False
Frame Number=222 Tailgate=False
Frame Number=223 Tailgate=False
Frame Number=224 Tailgate=False
Frame Number=225 Tailgate=False
Frame Number=226 Tailgate=False
Frame Number=227 Tailgate=False
Frame Number=228 Tailgate=False
Frame Number=229 Tailgate=False
Frame Number=230 Tailgate=False
Frame Number=231 Tailgate=False
Frame Number=232 Tailgate=False
Frame Number=233 Tailgate=False
Frame Number=234 Tailgate=False
Frame Number=235 Tailgate=False
Frame Number=236 Tailgate=False
Frame Number=237 Tailgate=False
Frame Number=238 Tailgate=False
Frame Number=239 Tailgate=False
Frame Number=240 Tailgate=False
Frame Number=241 Tailgate=False
Frame Number=242 Tailgate=False
Frame Number=243 Tailgate=False
Frame Number=244 Tailgate=False
Frame Number=245 Tailgate=False
Frame Number=246 Tailgate=False
Frame Number=247 Tailgate=False
Frame Number=248 Tailgate=False
Frame Number=249 Tailgate=False
Frame Number=250 Tailgate=False
Frame Number=251 Tailgate=False
Frame Number=252 Tailgate=False
Frame Number=253 Tailgate=False
Frame Number=254 Tailgate=False
Frame Number=255 Tailgate=False
Frame Number=256 Tailgate=False
Frame Number=257 Tailgate=False
Frame Number=258 Tailgate=False
Frame Number=259 Tailgate=False
Frame Number=260 Tailgate=False
Frame Number=261 Tailgate=False
Frame Number=262 Tailgate=False
Frame Number=263 Tailgate=False
Frame Number=264 Tailgate=False
Frame Number=265 Tailgate=False
Frame Number=266 Tailgate=False
Frame Number=267 Tailgate=False
Frame Number=268 Tailgate=False
Frame Number=269 Tailgate=False
Frame Number=270 Tailgate=False
Frame Number=271 Tailgate=False
Frame Number=272 Tailgate=False
Frame Number=273 Tailgate=False
Frame Number=274 Tailgate=False
Frame Number=275 Tailgate=False
Frame Number=276 Tailgate=False
Frame Number=277 Tailgate=False
Frame Number=278 Tailgate=False
Frame Number=279 Tailgate=False
Frame Number=280 Tailgate=False
Frame Number=281 Tailgate=False
Frame Number=282 Tailgate=False
Frame Number=283 Tailgate=False
Frame Number=284 Tailgate=False
Frame Number=285 Tailgate=False
Frame Number=286 Tailgate=False
Frame Number=287 Tailgate=False
Frame Number=288 Tailgate=False
Frame Number=289 Tailgate=False
Frame Number=290 Tailgate=False
Frame Number=291 Tailgate=False
Frame Number=292 Tailgate=False
Frame Number=293 Tailgate=False
Frame Number=294 Tailgate=False
Frame Number=295 Tailgate=False
Frame Number=296 Tailgate=False
Frame Number=297 Tailgate=False
Frame Number=298 Tailgate=False
Frame Number=299 Tailgate=False
Frame Number=300 Tailgate=False
Frame Number=301 Tailgate=False
Frame Number=302 Tailgate=False
Frame Number=303 Tailgate=False
Frame Number=304 Tailgate=False
Frame Number=305 Tailgate=False
Frame Number=306 Tailgate=False
Frame Number=307 Tailgate=False
Frame Number=308 Tailgate=False
Frame Number=309 Tailgate=False
Frame Number=310 Tailgate=False
Frame Number=311 Tailgate=False
Frame Number=312 Tailgate=False
Frame Number=313 Tailgate=False
Frame Number=314 Tailgate=False
Frame Number=315 Tailgate=False
Frame Number=316 Tailgate=False
Frame Number=317 Tailgate=False
Frame Number=318 Tailgate=False
Frame Number=319 Tailgate=False
Frame Number=320 Tailgate=False
Frame Number=321 Tailgate=False
Frame Number=322 Tailgate=False
Frame Number=323 Tailgate=False
Frame Number=324 Tailgate=False
Frame Number=325 Tailgate=False
Frame Number=326 Tailgate=False
Frame Number=327 Tailgate=False
Frame Number=328 Tailgate=False
Frame Number=329 Tailgate=False
Frame Number=330 Tailgate=False
Frame Number=331 Tailgate=False
Frame Number=332 Tailgate=False
Frame Number=333 Tailgate=False
Frame Number=334 Tailgate=False
Frame Number=335 Tailgate=False
Frame Number=336 Tailgate=False
Frame Number=337 Tailgate=False
Frame Number=338 Tailgate=False
Frame Number=339 Tailgate=False
Frame Number=340 Tailgate=False
Frame Number=341 Tailgate=False
Frame Number=342 Tailgate=False
Frame Number=343 Tailgate=False
Frame Number=344 Tailgate=False
Frame Number=345 Tailgate=False
Frame Number=346 Tailgate=False
Frame Number=347 Tailgate=False
Frame Number=348 Tailgate=False
Frame Number=349 Tailgate=False
Frame Number=350 Tailgate=False
Frame Number=351 Tailgate=False
Frame Number=352 Tailgate=False
Frame Number=353 Tailgate=False
Frame Number=354 Tailgate=False
Frame Number=355 Tailgate=False
Frame Number=356 Tailgate=False
Frame Number=357 Tailgate=False
Frame Number=358 Tailgate=False
Frame Number=359 Tailgate=False
Frame Number=360 Tailgate=False
Frame Number=361 Tailgate=False
Frame Number=362 Tailgate=False
Frame Number=363 Tailgate=False
Frame Number=364 Tailgate=False
Frame Number=365 Tailgate=False
Frame Number=366 Tailgate=False
Frame Number=367 Tailgate=False
Frame Number=368 Tailgate=False
Frame Number=369 Tailgate=False
Frame Number=370 Tailgate=False
Frame Number=371 Tailgate=False
Frame Number=372 Tailgate=False
Frame Number=373 Tailgate=False
Frame Number=374 Tailgate=False
Frame Number=375 Tailgate=False
Frame Number=376 Tailgate=False
Frame Number=377 Tailgate=False
Frame Number=378 Tailgate=False
Frame Number=379 Tailgate=False
Frame Number=380 Tailgate=False
Frame Number=381 Tailgate=False
Frame Number=382 Tailgate=False
Frame Number=383 Tailgate=False
Frame Number=384 Tailgate=False
Frame Number=385 Tailgate=False
Frame Number=386 Tailgate=False
Frame Number=387 Tailgate=False
Frame Number=388 Tailgate=False
Frame Number=389 Tailgate=False
Frame Number=390 Tailgate=False
Frame Number=391 Tailgate=False
Frame Number=392 Tailgate=False
Frame Number=393 Tailgate=False
Frame Number=394 Tailgate=False
Frame Number=395 Tailgate=False
Frame Number=396 Tailgate=False
Frame Number=397 Tailgate=False
Frame Number=398 Tailgate=False
Frame Number=399 Tailgate=False
Frame Number=400 Tailgate=False
Frame Number=401 Tailgate=False
Frame Number=402 Tailgate=False
Frame Number=403 Tailgate=False
Frame Number=404 Tailgate=False
Frame Number=405 Tailgate=False
Frame Number=406 Tailgate=False
Frame Number=407 Tailgate=False
Frame Number=408 Tailgate=False
Frame Number=409 Tailgate=False
Frame Number=410 Tailgate=False
Frame Number=411 Tailgate=False
Frame Number=412 Tailgate=False
Frame Number=413 Tailgate=False
Frame Number=414 Tailgate=False
Frame Number=415 Tailgate=False
Frame Number=416 Tailgate=False
Frame Number=417 Tailgate=False
Frame Number=418 Tailgate=False
Frame Number=419 Tailgate=False
Frame Number=420 Tailgate=False
Frame Number=421 Tailgate=False
Frame Number=422 Tailgate=False
Frame Number=423 Tailgate=False
Frame Number=424 Tailgate=False
Frame Number=425 Tailgate=False
Frame Number=426 Tailgate=False
Frame Number=427 Tailgate=False
Frame Number=428 Tailgate=False
Frame Number=429 Tailgate=False
Frame Number=430 Tailgate=False
Frame Number=431 Tailgate=False
Frame Number=432 Tailgate=False
Frame Number=433 Tailgate=False
Frame Number=434 Tailgate=False
Frame Number=435 Tailgate=False
Frame Number=436 Tailgate=False
Frame Number=437 Tailgate=False
Frame Number=438 Tailgate=False
Frame Number=439 Tailgate=False
Frame Number=440 Tailgate=False
Frame Number=441 Tailgate=False
Frame Number=442 Tailgate=False
Frame Number=443 Tailgate=False
Frame Number=444 Tailgate=False
Frame Number=445 Tailgate=False
Frame Number=446 Tailgate=False
Frame Number=447 Tailgate=False
Frame Number=448 Tailgate=False
Frame Number=449 Tailgate=False
Frame Number=450 Tailgate=False
Frame Number=451 Tailgate=False
Frame Number=452 Tailgate=False
Frame Number=453 Tailgate=False
Frame Number=454 Tailgate=False
Frame Number=455 Tailgate=False
Frame Number=456 Tailgate=False
Frame Number=457 Tailgate=False
Frame Number=458 Tailgate=False
Frame Number=459 Tailgate=False
Frame Number=460 Tailgate=False
Frame Number=461 Tailgate=False
Frame Number=462 Tailgate=False
Frame Number=463 Tailgate=False
Frame Number=464 Tailgate=False
Frame Number=465 Tailgate=False
Frame Number=466 Tailgate=False
Frame Number=467 Tailgate=False
Frame Number=468 Tailgate=False
Frame Number=469 Tailgate=False
Frame Number=470 Tailgate=False
Frame Number=471 Tailgate=False
Frame Number=472 Tailgate=False
Frame Number=473 Tailgate=False
Frame Number=474 Tailgate=False
Frame Number=475 Tailgate=False
Frame Number=476 Tailgate=False
Frame Number=477 Tailgate=False
Frame Number=478 Tailgate=False
Frame Number=479 Tailgate=False
Frame Number=480 Tailgate=False
Frame Number=481 Tailgate=False
Frame Number=482 Tailgate=False
Frame Number=483 Tailgate=False
Frame Number=484 Tailgate=False
Frame Number=485 Tailgate=False
Frame Number=486 Tailgate=False
Frame Number=487 Tailgate=False
Frame Number=488 Tailgate=False
Frame Number=489 Tailgate=False
Frame Number=490 Tailgate=False
Frame Number=491 Tailgate=False
Frame Number=492 Tailgate=False
Frame Number=493 Tailgate=False
Frame Number=494 Tailgate=False
Frame Number=495 Tailgate=False
Frame Number=496 Tailgate=False
Frame Number=497 Tailgate=False
Frame Number=498 Tailgate=False
Frame Number=499 Tailgate=False
Frame Number=500 Tailgate=False
Frame Number=501 Tailgate=False
Frame Number=502 Tailgate=False
Frame Number=503 Tailgate=False
Frame Number=504 Tailgate=False
Frame Number=505 Tailgate=False
Frame Number=506 Tailgate=False
Frame Number=507 Tailgate=False
Frame Number=508 Tailgate=False
Frame Number=509 Tailgate=False
Frame Number=510 Tailgate=False
Frame Number=511 Tailgate=False
Frame Number=512 Tailgate=False
Frame Number=513 Tailgate=False
Frame Number=514 Tailgate=False
Frame Number=515 Tailgate=False
Frame Number=516 Tailgate=False
Frame Number=517 Tailgate=False
Frame Number=518 Tailgate=False
Frame Number=519 Tailgate=False
Frame Number=520 Tailgate=False
Frame Number=521 Tailgate=False
Frame Number=522 Tailgate=False
Frame Number=523 Tailgate=False
Frame Number=524 Tailgate=False
Frame Number=525 Tailgate=False
Frame Number=526 Tailgate=False
Frame Number=527 Tailgate=False
Frame Number=528 Tailgate=False
Frame Number=529 Tailgate=False
Frame Number=530 Tailgate=False
Frame Number=531 Tailgate=False
Frame Number=532 Tailgate=False
Frame Number=533 Tailgate=False
Frame Number=534 Tailgate=False
Frame Number=535 Tailgate=False
Frame Number=536 Tailgate=False
Frame Number=537 Tailgate=False
Frame Number=538 Tailgate=False
Frame Number=539 Tailgate=False
Frame Number=540 Tailgate=False
Frame Number=541 Tailgate=False
Frame Number=542 Tailgate=False
Frame Number=543 Tailgate=False
Frame Number=544 Tailgate=False
Frame Number=545 Tailgate=False
Frame Number=546 Tailgate=False
Frame Number=547 Tailgate=False
Frame Number=548 Tailgate=False
Frame Number=549 Tailgate=False
Frame Number=550 Tailgate=False
Frame Number=551 Tailgate=False
Frame Number=552 Tailgate=False
Frame Number=553 Tailgate=False
Frame Number=554 Tailgate=False
Frame Number=555 Tailgate=False
Frame Number=556 Tailgate=False
Frame Number=557 Tailgate=False
Frame Number=558 Tailgate=False
Frame Number=559 Tailgate=False
Frame Number=560 Tailgate=False
Frame Number=561 Tailgate=False
Frame Number=562 Tailgate=False
Frame Number=563 Tailgate=False
Frame Number=564 Tailgate=False
Frame Number=565 Tailgate=False
Frame Number=566 Tailgate=False
Frame Number=567 Tailgate=False
Frame Number=568 Tailgate=False
Frame Number=569 Tailgate=False
Frame Number=570 Tailgate=False
Frame Number=571 Tailgate=False
Frame Number=572 Tailgate=False
Frame Number=573 Tailgate=False
Frame Number=574 Tailgate=False
Frame Number=575 Tailgate=False
Frame Number=576 Tailgate=False
Frame Number=577 Tailgate=False
Frame Number=578 Tailgate=False
Frame Number=579 Tailgate=False
Frame Number=580 Tailgate=False
Frame Number=581 Tailgate=False
Frame Number=582 Tailgate=False
Frame Number=583 Tailgate=False
Frame Number=584 Tailgate=False
Frame Number=585 Tailgate=False
Frame Number=586 Tailgate=False
Frame Number=587 Tailgate=False
Frame Number=588 Tailgate=False
Frame Number=589 Tailgate=False
Frame Number=590 Tailgate=False
Frame Number=591 Tailgate=False
Frame Number=592 Tailgate=False
Frame Number=593 Tailgate=False
Frame Number=594 Tailgate=False
Frame Number=595 Tailgate=False
Frame Number=596 Tailgate=False
Frame Number=597 Tailgate=False
Frame Number=598 Tailgate=False
Frame Number=599 Tailgate=False
Frame Number=600 Tailgate=False
Frame Number=601 Tailgate=False
Frame Number=602 Tailgate=False
Frame Number=603 Tailgate=False
Frame Number=604 Tailgate=False
Frame Number=605 Tailgate=False
Frame Number=606 Tailgate=False
Frame Number=607 Tailgate=False
Frame Number=608 Tailgate=False
Frame Number=609 Tailgate=False
Frame Number=610 Tailgate=False
Frame Number=611 Tailgate=False
Frame Number=612 Tailgate=False
Frame Number=613 Tailgate=False
Frame Number=614 Tailgate=False
Frame Number=615 Tailgate=False
Frame Number=616 Tailgate=False
Frame Number=617 Tailgate=False
Frame Number=618 Tailgate=False
Frame Number=619 Tailgate=False
Frame Number=620 Tailgate=False
Frame Number=621 Tailgate=False
Frame Number=622 Tailgate=False
Frame Number=623 Tailgate=False
Frame Number=624 Tailgate=False
Frame Number=625 Tailgate=False
Frame Number=626 Tailgate=False
Frame Number=627 Tailgate=False
Frame Number=628 Tailgate=False
Frame Number=629 Tailgate=False
Frame Number=630 Tailgate=False
Frame Number=631 Tailgate=False
Frame Number=632 Tailgate=False
Frame Number=633 Tailgate=False
Frame Number=634 Tailgate=False
Frame Number=635 Tailgate=False
Frame Number=636 Tailgate=False
Frame Number=637 Tailgate=False
Frame Number=638 Tailgate=False
Frame Number=639 Tailgate=False
Frame Number=640 Tailgate=False
Frame Number=641 Tailgate=False
Frame Number=642 Tailgate=False
Frame Number=643 Tailgate=False
Frame Number=644 Tailgate=False
Frame Number=645 Tailgate=False
Frame Number=646 Tailgate=False
Frame Number=647 Tailgate=False
Frame Number=648 Tailgate=False
Frame Number=649 Tailgate=False
Frame Number=650 Tailgate=False
Frame Number=651 Tailgate=False
Frame Number=652 Tailgate=False
Frame Number=653 Tailgate=False
Frame Number=654 Tailgate=False
Frame Number=655 Tailgate=False
Frame Number=656 Tailgate=False
Frame Number=657 Tailgate=False
Frame Number=658 Tailgate=False
Frame Number=659 Tailgate=False
Frame Number=660 Tailgate=False
Frame Number=661 Tailgate=False
Frame Number=662 Tailgate=False
Frame Number=663 Tailgate=False
Frame Number=664 Tailgate=False
Frame Number=665 Tailgate=False
Frame Number=666 Tailgate=False
Frame Number=667 Tailgate=False
Frame Number=668 Tailgate=False
Frame Number=669 Tailgate=False
Frame Number=670 Tailgate=False
Frame Number=671 Tailgate=False
Frame Number=672 Tailgate=False
Frame Number=673 Tailgate=False
Frame Number=674 Tailgate=False
Frame Number=675 Tailgate=False
Frame Number=676 Tailgate=False
Frame Number=677 Tailgate=False
Frame Number=678 Tailgate=False
Frame Number=679 Tailgate=False
Frame Number=680 Tailgate=False
Frame Number=681 Tailgate=False
Frame Number=682 Tailgate=False
Frame Number=683 Tailgate=False
Frame Number=684 Tailgate=False
Frame Number=685 Tailgate=False
Frame Number=686 Tailgate=False
Frame Number=687 Tailgate=False
Frame Number=688 Tailgate=False
Frame Number=689 Tailgate=False
Frame Number=690 Tailgate=False
Frame Number=691 Tailgate=False
Frame Number=692 Tailgate=False
Frame Number=693 Tailgate=False
Frame Number=694 Tailgate=False
Frame Number=695 Tailgate=False
Frame Number=696 Tailgate=False
Frame Number=697 Tailgate=False
Frame Number=698 Tailgate=False
Frame Number=699 Tailgate=False
Frame Number=700 Tailgate=False
Frame Number=701 Tailgate=False
Frame Number=702 Tailgate=False
Frame Number=703 Tailgate=False
Frame Number=704 Tailgate=False
Frame Number=705 Tailgate=False
Frame Number=706 Tailgate=False
Frame Number=707 Tailgate=False
Frame Number=708 Tailgate=False
Frame Number=709 Tailgate=False
Frame Number=710 Tailgate=False
Frame Number=711 Tailgate=False
Frame Number=712 Tailgate=False
Frame Number=713 Tailgate=False
Frame Number=714 Tailgate=False
Frame Number=715 Tailgate=False
Frame Number=716 Tailgate=False
Frame Number=717 Tailgate=False
Frame Number=718 Tailgate=False
Frame Number=719 Tailgate=False
Frame Number=720 Tailgate=False
Frame Number=721 Tailgate=False
Frame Number=722 Tailgate=False
Frame Number=723 Tailgate=False
Frame Number=724 Tailgate=False
Frame Number=725 Tailgate=False
Frame Number=726 Tailgate=False
Frame Number=727 Tailgate=False
Frame Number=728 Tailgate=False
Frame Number=729 Tailgate=False
Frame Number=730 Tailgate=False
Frame Number=731 Tailgate=False
Frame Number=732 Tailgate=False
Frame Number=733 Tailgate=False
Frame Number=734 Tailgate=False
Frame Number=735 Tailgate=False
Frame Number=736 Tailgate=False
Frame Number=737 Tailgate=False
Frame Number=738 Tailgate=False
Frame Number=739 Tailgate=False
Frame Number=740 Tailgate=False
Frame Number=741 Tailgate=False
Frame Number=742 Tailgate=False
Frame Number=743 Tailgate=False
Frame Number=744 Tailgate=False
Frame Number=745 Tailgate=False
Frame Number=746 Tailgate=False
Frame Number=747 Tailgate=False
Frame Number=748 Tailgate=False
Frame Number=749 Tailgate=False
Frame Number=750 Tailgate=False
Frame Number=751 Tailgate=False
Frame Number=752 Tailgate=False
Frame Number=753 Tailgate=False
Frame Number=754 Tailgate=False
Frame Number=755 Tailgate=False
Frame Number=756 Tailgate=False
Frame Number=757 Tailgate=False
Frame Number=758 Tailgate=False
Frame Number=759 Tailgate=False
Frame Number=760 Tailgate=False
Frame Number=761 Tailgate=False
Frame Number=762 Tailgate=False
Frame Number=763 Tailgate=False
Frame Number=764 Tailgate=False
Frame Number=765 Tailgate=False
Frame Number=766 Tailgate=False
Frame Number=767 Tailgate=False
Frame Number=768 Tailgate=False
Frame Number=769 Tailgate=False
Frame Number=770 Tailgate=False
Frame Number=771 Tailgate=False
Frame Number=772 Tailgate=False
Frame Number=773 Tailgate=False
Frame Number=774 Tailgate=False
Frame Number=775 Tailgate=False
Frame Number=776 Tailgate=False
Frame Number=777 Tailgate=False
Frame Number=778 Tailgate=False
Frame Number=779 Tailgate=False
Frame Number=780 Tailgate=False
Frame Number=781 Tailgate=False
Frame Number=782 Tailgate=False
Frame Number=783 Tailgate=False
Frame Number=784 Tailgate=False
Frame Number=785 Tailgate=False
Frame Number=786 Tailgate=False
Frame Number=787 Tailgate=False
Frame Number=788 Tailgate=False
Frame Number=789 Tailgate=False
Frame Number=790 Tailgate=False
Frame Number=791 Tailgate=False
Frame Number=792 Tailgate=False
Frame Number=793 Tailgate=False
Frame Number=794 Tailgate=False
Frame Number=795 Tailgate=False
Frame Number=796 Tailgate=False
Frame Number=797 Tailgate=False
Frame Number=798 Tailgate=False
Frame Number=799 Tailgate=False
Frame Number=800 Tailgate=False
Frame Number=801 Tailgate=False
Frame Number=802 Tailgate=False
Frame Number=803 Tailgate=False
Frame Number=804 Tailgate=False
Frame Number=805 Tailgate=False
Frame Number=806 Tailgate=False
Frame Number=807 Tailgate=False
Frame Number=808 Tailgate=False
Frame Number=809 Tailgate=False
Frame Number=810 Tailgate=False
Frame Number=811 Tailgate=False
Frame Number=812 Tailgate=False
Frame Number=813 Tailgate=False
Frame Number=814 Tailgate=False
Frame Number=815 Tailgate=False
Frame Number=816 Tailgate=False
Frame Number=817 Tailgate=False
Frame Number=818 Tailgate=False
Frame Number=819 Tailgate=False
Frame Number=820 Tailgate=False
Frame Number=821 Tailgate=False
Frame Number=822 Tailgate=False
Frame Number=823 Tailgate=False
Frame Number=824 Tailgate=False
Frame Number=825 Tailgate=False
Frame Number=826 Tailgate=False
Frame Number=827 Tailgate=False
Frame Number=828 Tailgate=False
Frame Number=829 Tailgate=False
Frame Number=830 Tailgate=False
Frame Number=831 Tailgate=False
Frame Number=832 Tailgate=False
Frame Number=833 Tailgate=False
Frame Number=834 Tailgate=False
Frame Number=835 Tailgate=False
Frame Number=836 Tailgate=False
Frame Number=837 Tailgate=False
Frame Number=838 Tailgate=False
Frame Number=839 Tailgate=False
Frame Number=840 Tailgate=False
Frame Number=841 Tailgate=False
Frame Number=842 Tailgate=False
Frame Number=843 Tailgate=False
Frame Number=844 Tailgate=False
Frame Number=845 Tailgate=False
Frame Number=846 Tailgate=False
Frame Number=847 Tailgate=False
Frame Number=848 Tailgate=False
Frame Number=849 Tailgate=False
Frame Number=850 Tailgate=False
Frame Number=851 Tailgate=False
Frame Number=852 Tailgate=False
Frame Number=853 Tailgate=False
Frame Number=854 Tailgate=False
Frame Number=855 Tailgate=False
Frame Number=856 Tailgate=False
Frame Number=857 Tailgate=False
Frame Number=858 Tailgate=False
Frame Number=859 Tailgate=False
Frame Number=860 Tailgate=False
Frame Number=861 Tailgate=False
Frame Number=862 Tailgate=False
Frame Number=863 Tailgate=False
Frame Number=864 Tailgate=False
Frame Number=865 Tailgate=False
Frame Number=866 Tailgate=False
Frame Number=867 Tailgate=False
Frame Number=868 Tailgate=False
Frame Number=869 Tailgate=False
Frame Number=870 Tailgate=False
Frame Number=871 Tailgate=False
Frame Number=872 Tailgate=False
Frame Number=873 Tailgate=False
Frame Number=874 Tailgate=False
Frame Number=875 Tailgate=False
Frame Number=876 Tailgate=False
Frame Number=877 Tailgate=False
Frame Number=878 Tailgate=False
Frame Number=879 Tailgate=False
Frame Number=880 Tailgate=False
Frame Number=881 Tailgate=False
Frame Number=882 Tailgate=False
Frame Number=883 Tailgate=False
Frame Number=884 Tailgate=False
Frame Number=885 Tailgate=False
Frame Number=886 Tailgate=False
Frame Number=887 Tailgate=False
Frame Number=888 Tailgate=False
Frame Number=889 Tailgate=False
Frame Number=890 Tailgate=False
Frame Number=891 Tailgate=False
Frame Number=892 Tailgate=False
Frame Number=893 Tailgate=False
Frame Number=894 Tailgate=False
Frame Number=895 Tailgate=False
Frame Number=896 Tailgate=False
Frame Number=897 Tailgate=False
Frame Number=898 Tailgate=False
Frame Number=899 Tailgate=False
Frame Number=900 Tailgate=False
Frame Number=901 Tailgate=False
Frame Number=902 Tailgate=False
Frame Number=903 Tailgate=False
Frame Number=904 Tailgate=False
Frame Number=905 Tailgate=False
Frame Number=906 Tailgate=False
Frame Number=907 Tailgate=False
Frame Number=908 Tailgate=False
Frame Number=909 Tailgate=False
Frame Number=910 Tailgate=False
Frame Number=911 Tailgate=False
Frame Number=912 Tailgate=False
Frame Number=913 Tailgate=False
Frame Number=914 Tailgate=False
Frame Number=915 Tailgate=False
Frame Number=916 Tailgate=False
Frame Number=917 Tailgate=False
Frame Number=918 Tailgate=False
Frame Number=919 Tailgate=False
Frame Number=920 Tailgate=False
Frame Number=921 Tailgate=False
Frame Number=922 Tailgate=False
Frame Number=923 Tailgate=False
Frame Number=924 Tailgate=False
Frame Number=925 Tailgate=False
Frame Number=926 Tailgate=True
Frame Number=927 Tailgate=True
Frame Number=928 Tailgate=False
Frame Number=929 Tailgate=False
Frame Number=930 Tailgate=False
Frame Number=931 Tailgate=False
Frame Number=932 Tailgate=False
Frame Number=933 Tailgate=False
Frame Number=934 Tailgate=False
Frame Number=935 Tailgate=False
Frame Number=936 Tailgate=False
Frame Number=937 Tailgate=False
Frame Number=938 Tailgate=False
Frame Number=939 Tailgate=False
Frame Number=940 Tailgate=False
Frame Number=941 Tailgate=False
Frame Number=942 Tailgate=False
Frame Number=943 Tailgate=False
Frame Number=944 Tailgate=False
Frame Number=945 Tailgate=False
Frame Number=946 Tailgate=False
Frame Number=947 Tailgate=False
Frame Number=948 Tailgate=False
Frame Number=949 Tailgate=False
Frame Number=950 Tailgate=False
Frame Number=951 Tailgate=False
Frame Number=952 Tailgate=False
Frame Number=953 Tailgate=False
Frame Number=954 Tailgate=False
Frame Number=955 Tailgate=False
Frame Number=956 Tailgate=False
Frame Number=957 Tailgate=False
Frame Number=958 Tailgate=False
Frame Number=959 Tailgate=False
Frame Number=960 Tailgate=False
Frame Number=961 Tailgate=False
Frame Number=962 Tailgate=False
Frame Number=963 Tailgate=False
Frame Number=964 Tailgate=False
Frame Number=965 Tailgate=False
Frame Number=966 Tailgate=False
Frame Number=967 Tailgate=False
Frame Number=968 Tailgate=False
Frame Number=969 Tailgate=False
Frame Number=970 Tailgate=False
Frame Number=971 Tailgate=False
Frame Number=972 Tailgate=False
Frame Number=973 Tailgate=False
Frame Number=974 Tailgate=False
Frame Number=975 Tailgate=False
Frame Number=976 Tailgate=False
Frame Number=977 Tailgate=False
Frame Number=978 Tailgate=False
Frame Number=979 Tailgate=False
Frame Number=980 Tailgate=False
Frame Number=981 Tailgate=False
Frame Number=982 Tailgate=False
Frame Number=983 Tailgate=False
Frame Number=984 Tailgate=False
Frame Number=985 Tailgate=False
Frame Number=986 Tailgate=False
Frame Number=987 Tailgate=False
Frame Number=988 Tailgate=False
Frame Number=989 Tailgate=False
Frame Number=990 Tailgate=False
Frame Number=991 Tailgate=False
Frame Number=992 Tailgate=False
Frame Number=993 Tailgate=False
Frame Number=994 Tailgate=False
Frame Number=995 Tailgate=False
Frame Number=996 Tailgate=False
Frame Number=997 Tailgate=False
Frame Number=998 Tailgate=False
Frame Number=999 Tailgate=False
Frame Number=1000 Tailgate=False
Frame Number=1001 Tailgate=False
Frame Number=1002 Tailgate=False
Frame Number=1003 Tailgate=False
Frame Number=1004 Tailgate=False
Frame Number=1005 Tailgate=False
Frame Number=1006 Tailgate=False
Frame Number=1007 Tailgate=False
Frame Number=1008 Tailgate=False
Frame Number=1009 Tailgate=False
Frame Number=1010 Tailgate=False
Frame Number=1011 Tailgate=False
Frame Number=1012 Tailgate=False
Frame Number=1013 Tailgate=False
Frame Number=1014 Tailgate=False
Frame Number=1015 Tailgate=False
Frame Number=1016 Tailgate=False
Frame Number=1017 Tailgate=False
Frame Number=1018 Tailgate=False
Frame Number=1019 Tailgate=False
Frame Number=1020 Tailgate=False
Frame Number=1021 Tailgate=False
Frame Number=1022 Tailgate=False
Frame Number=1023 Tailgate=False
Frame Number=1024 Tailgate=False
Frame Number=1025 Tailgate=False
Frame Number=1026 Tailgate=False
Frame Number=1027 Tailgate=False
Frame Number=1028 Tailgate=False
Frame Number=1029 Tailgate=False
Frame Number=1030 Tailgate=False
Frame Number=1031 Tailgate=False
Frame Number=1032 Tailgate=False
Frame Number=1033 Tailgate=False
Frame Number=1034 Tailgate=False
Frame Number=1035 Tailgate=False
Frame Number=1036 Tailgate=False
Frame Number=1037 Tailgate=False
Frame Number=1038 Tailgate=False
Frame Number=1039 Tailgate=False
Frame Number=1040 Tailgate=False
Frame Number=1041 Tailgate=False
Frame Number=1042 Tailgate=False
Frame Number=1043 Tailgate=False
Frame Number=1044 Tailgate=False
Frame Number=1045 Tailgate=False
Frame Number=1046 Tailgate=False
Frame Number=1047 Tailgate=False
Frame Number=1048 Tailgate=False
Frame Number=1049 Tailgate=False
Frame Number=1050 Tailgate=False
Frame Number=1051 Tailgate=False
Frame Number=1052 Tailgate=False
Frame Number=1053 Tailgate=False
Frame Number=1054 Tailgate=False
Frame Number=1055 Tailgate=False
Frame Number=1056 Tailgate=False
Frame Number=1057 Tailgate=False
Frame Number=1058 Tailgate=False
Frame Number=1059 Tailgate=False
Frame Number=1060 Tailgate=False
Frame Number=1061 Tailgate=False
Frame Number=1062 Tailgate=False
Frame Number=1063 Tailgate=False
Frame Number=1064 Tailgate=False
Frame Number=1065 Tailgate=False
Frame Number=1066 Tailgate=False
Frame Number=1067 Tailgate=False
Frame Number=1068 Tailgate=False
Frame Number=1069 Tailgate=False
Frame Number=1070 Tailgate=False
Frame Number=1071 Tailgate=False
Frame Number=1072 Tailgate=False
Frame Number=1073 Tailgate=False
Frame Number=1074 Tailgate=False
Frame Number=1075 Tailgate=False
Frame Number=1076 Tailgate=False
Frame Number=1077 Tailgate=False
Frame Number=1078 Tailgate=False
Frame Number=1079 Tailgate=False
Frame Number=1080 Tailgate=False
Frame Number=1081 Tailgate=False
Frame Number=1082 Tailgate=False
Frame Number=1083 Tailgate=False
Frame Number=1084 Tailgate=False
Frame Number=1085 Tailgate=False
Frame Number=1086 Tailgate=False
Frame Number=1087 Tailgate=False
Frame Number=1088 Tailgate=False
Frame Number=1089 Tailgate=False
Frame Number=1090 Tailgate=False
Frame Number=1091 Tailgate=False
Frame Number=1092 Tailgate=False
Frame Number=1093 Tailgate=False
Frame Number=1094 Tailgate=False
Frame Number=1095 Tailgate=False
Frame Number=1096 Tailgate=False
Frame Number=1097 Tailgate=False
Frame Number=1098 Tailgate=False
Frame Number=1099 Tailgate=False
Frame Number=1100 Tailgate=True
Frame Number=1101 Tailgate=True
Frame Number=1102 Tailgate=False
Frame Number=1103 Tailgate=False
Frame Number=1104 Tailgate=False
Frame Number=1105 Tailgate=False
Frame Number=1106 Tailgate=False
Frame Number=1107 Tailgate=False
Frame Number=1108 Tailgate=False
Frame Number=1109 Tailgate=False
Frame Number=1110 Tailgate=False
Frame Number=1111 Tailgate=False
Frame Number=1112 Tailgate=False
Frame Number=1113 Tailgate=False
Frame Number=1114 Tailgate=False
Frame Number=1115 Tailgate=False
Frame Number=1116 Tailgate=False
Frame Number=1117 Tailgate=False
Frame Number=1118 Tailgate=False
Frame Number=1119 Tailgate=False
Frame Number=1120 Tailgate=False
Frame Number=1121 Tailgate=False
Frame Number=1122 Tailgate=False
Frame Number=1123 Tailgate=False
Frame Number=1124 Tailgate=False
Frame Number=1125 Tailgate=False
Frame Number=1126 Tailgate=False
Frame Number=1127 Tailgate=False
Frame Number=1128 Tailgate=False
Frame Number=1129 Tailgate=False
Frame Number=1130 Tailgate=True
Frame Number=1131 Tailgate=True
Frame Number=1132 Tailgate=True
Frame Number=1133 Tailgate=True
Frame Number=1134 Tailgate=True
Frame Number=1135 Tailgate=True
Frame Number=1136 Tailgate=True
Frame Number=1137 Tailgate=True
Frame Number=1138 Tailgate=True
Frame Number=1139 Tailgate=True
Frame Number=1140 Tailgate=True
Frame Number=1141 Tailgate=True
Frame Number=1142 Tailgate=False
Frame Number=1143 Tailgate=False
Frame Number=1144 Tailgate=False
Frame Number=1145 Tailgate=False
Frame Number=1146 Tailgate=False
Frame Number=1147 Tailgate=False
Frame Number=1148 Tailgate=False
Frame Number=1149 Tailgate=False
Frame Number=1150 Tailgate=True
Frame Number=1151 Tailgate=True
Frame Number=1152 Tailgate=False
Frame Number=1153 Tailgate=False
Frame Number=1154 Tailgate=False
Frame Number=1155 Tailgate=False
Frame Number=1156 Tailgate=False
Frame Number=1157 Tailgate=False
Frame Number=1158 Tailgate=True
Frame Number=1159 Tailgate=True
Frame Number=1160 Tailgate=False
Frame Number=1161 Tailgate=False
Frame Number=1162 Tailgate=False
Frame Number=1163 Tailgate=False
Frame Number=1164 Tailgate=True
Frame Number=1165 Tailgate=True
Frame Number=1166 Tailgate=True
Frame Number=1167 Tailgate=True
Frame Number=1168 Tailgate=True
Frame Number=1169 Tailgate=True
Frame Number=1170 Tailgate=True
Frame Number=1171 Tailgate=True
Frame Number=1172 Tailgate=True
Frame Number=1173 Tailgate=True
Frame Number=1174 Tailgate=True
Frame Number=1175 Tailgate=True
Frame Number=1176 Tailgate=True
Frame Number=1177 Tailgate=True
Frame Number=1178 Tailgate=True
Frame Number=1179 Tailgate=True
Frame Number=1180 Tailgate=True
Frame Number=1181 Tailgate=True
Frame Number=1182 Tailgate=True
Frame Number=1183 Tailgate=True
Frame Number=1184 Tailgate=True
Frame Number=1185 Tailgate=True
Frame Number=1186 Tailgate=True
Frame Number=1187 Tailgate=True
Frame Number=1188 Tailgate=True
Frame Number=1189 Tailgate=True
Frame Number=1190 Tailgate=True
Frame Number=1191 Tailgate=True
Frame Number=1192 Tailgate=True
Frame Number=1193 Tailgate=True
Frame Number=1194 Tailgate=True
Frame Number=1195 Tailgate=True
Frame Number=1196 Tailgate=True
Frame Number=1197 Tailgate=True
Frame Number=1198 Tailgate=True
Frame Number=1199 Tailgate=True
Frame Number=1200 Tailgate=False
Frame Number=1201 Tailgate=False
Frame Number=1202 Tailgate=True
Frame Number=1203 Tailgate=True
Frame Number=1204 Tailgate=False
Frame Number=1205 Tailgate=True
Frame Number=1206 Tailgate=True
Frame Number=1207 Tailgate=False
Frame Number=1208 Tailgate=True
Frame Number=1209 Tailgate=True
Frame Number=1210 Tailgate=True
Frame Number=1211 Tailgate=True
Frame Number=1212 Tailgate=True
Frame Number=1213 Tailgate=True
Frame Number=1214 Tailgate=True
Frame Number=1215 Tailgate=True
Frame Number=1216 Tailgate=True
Frame Number=1217 Tailgate=True
Frame Number=1218 Tailgate=True
Frame Number=1219 Tailgate=True
Frame Number=1220 Tailgate=True
Frame Number=1221 Tailgate=True
Frame Number=1222 Tailgate=True
Frame Number=1223 Tailgate=True
Frame Number=1224 Tailgate=True
Frame Number=1225 Tailgate=True
Frame Number=1226 Tailgate=True
Frame Number=1227 Tailgate=True
Frame Number=1228 Tailgate=True
Frame Number=1229 Tailgate=True
Frame Number=1230 Tailgate=True
Frame Number=1231 Tailgate=True
Frame Number=1232 Tailgate=True
Frame Number=1233 Tailgate=True
Frame Number=1234 Tailgate=True
Frame Number=1235 Tailgate=True
Frame Number=1236 Tailgate=True
Frame Number=1237 Tailgate=True
Frame Number=1238 Tailgate=True
Frame Number=1239 Tailgate=True
Frame Number=1240 Tailgate=True
Frame Number=1241 Tailgate=True
Frame Number=1242 Tailgate=True
Frame Number=1243 Tailgate=True
Frame Number=1244 Tailgate=True
Frame Number=1245 Tailgate=True
Frame Number=1246 Tailgate=True
Frame Number=1247 Tailgate=True
Frame Number=1248 Tailgate=True
Frame Number=1249 Tailgate=True
Frame Number=1250 Tailgate=True
Frame Number=1251 Tailgate=True
Frame Number=1252 Tailgate=True
Frame Number=1253 Tailgate=True
Frame Number=1254 Tailgate=True
Frame Number=1255 Tailgate=True
Frame Number=1256 Tailgate=True
Frame Number=1257 Tailgate=True
Frame Number=1258 Tailgate=True
Frame Number=1259 Tailgate=True
Frame Number=1260 Tailgate=True
Frame Number=1261 Tailgate=True
Frame Number=1262 Tailgate=True
Frame Number=1263 Tailgate=True
Frame Number=1264 Tailgate=True
Frame Number=1265 Tailgate=True
Frame Number=1266 Tailgate=True
Frame Number=1267 Tailgate=True
Frame Number=1268 Tailgate=True
Frame Number=1269 Tailgate=True
Frame Number=1270 Tailgate=True
Frame Number=1271 Tailgate=True
Frame Number=1272 Tailgate=True
Frame Number=1273 Tailgate=True
Frame Number=1274 Tailgate=True
Frame Number=1275 Tailgate=True
Frame Number=1276 Tailgate=True
Frame Number=1277 Tailgate=True
Frame Number=1278 Tailgate=True
Frame Number=1279 Tailgate=True
Frame Number=1280 Tailgate=True
Frame Number=1281 Tailgate=True
Frame Number=1282 Tailgate=True
Frame Number=1283 Tailgate=True
Frame Number=1284 Tailgate=True
Frame Number=1285 Tailgate=True
Frame Number=1286 Tailgate=True
Frame Number=1287 Tailgate=True
Frame Number=1288 Tailgate=True
Frame Number=1289 Tailgate=True
Frame Number=1290 Tailgate=True
Frame Number=1291 Tailgate=True
Frame Number=1292 Tailgate=True
Frame Number=1293 Tailgate=True
Frame Number=1294 Tailgate=True
Frame Number=1295 Tailgate=True
Frame Number=1296 Tailgate=True
Frame Number=1297 Tailgate=True
Frame Number=1298 Tailgate=True
Frame Number=1299 Tailgate=True
Frame Number=1300 Tailgate=True
Frame Number=1301 Tailgate=True
Frame Number=1302 Tailgate=True
Frame Number=1303 Tailgate=True
Frame Number=1304 Tailgate=True
Frame Number=1305 Tailgate=True
Frame Number=1306 Tailgate=True
Frame Number=1307 Tailgate=True
Frame Number=1308 Tailgate=True
Frame Number=1309 Tailgate=True
Frame Number=1310 Tailgate=True
Frame Number=1311 Tailgate=True
Frame Number=1312 Tailgate=True
Frame Number=1313 Tailgate=True
Frame Number=1314 Tailgate=True
Frame Number=1315 Tailgate=True
Frame Number=1316 Tailgate=True
Frame Number=1317 Tailgate=True
Frame Number=1318 Tailgate=True
Frame Number=1319 Tailgate=True
Frame Number=1320 Tailgate=True
Frame Number=1321 Tailgate=True
Frame Number=1322 Tailgate=True
Frame Number=1323 Tailgate=True
Frame Number=1324 Tailgate=True
Frame Number=1325 Tailgate=True
Frame Number=1326 Tailgate=True
Frame Number=1327 Tailgate=False
Frame Number=1328 Tailgate=True
Frame Number=1329 Tailgate=True
Frame Number=1330 Tailgate=True
Frame Number=1331 Tailgate=True
Frame Number=1332 Tailgate=True
Frame Number=1333 Tailgate=True
Frame Number=1334 Tailgate=True
Frame Number=1335 Tailgate=True
Frame Number=1336 Tailgate=True
Frame Number=1337 Tailgate=True
Frame Number=1338 Tailgate=True
Frame Number=1339 Tailgate=True
Frame Number=1340 Tailgate=True
Frame Number=1341 Tailgate=True
Frame Number=1342 Tailgate=True
Frame Number=1343 Tailgate=True
Frame Number=1344 Tailgate=True
Frame Number=1345 Tailgate=True
Frame Number=1346 Tailgate=True
Frame Number=1347 Tailgate=True
Frame Number=1348 Tailgate=True
Frame Number=1349 Tailgate=True
Frame Number=1350 Tailgate=True
Frame Number=1351 Tailgate=True
Frame Number=1352 Tailgate=True
Frame Number=1353 Tailgate=True
Frame Number=1354 Tailgate=True
Frame Number=1355 Tailgate=True
Frame Number=1356 Tailgate=True
Frame Number=1357 Tailgate=True
Frame Number=1358 Tailgate=True
Frame Number=1359 Tailgate=True
Frame Number=1360 Tailgate=True
Frame Number=1361 Tailgate=True
Frame Number=1362 Tailgate=True
Frame Number=1363 Tailgate=True
Frame Number=1364 Tailgate=True
Frame Number=1365 Tailgate=True
Frame Number=1366 Tailgate=True
Frame Number=1367 Tailgate=True
Frame Number=1368 Tailgate=True
Frame Number=1369 Tailgate=True
Frame Number=1370 Tailgate=True
Frame Number=1371 Tailgate=True
Frame Number=1372 Tailgate=True
Frame Number=1373 Tailgate=True
Frame Number=1374 Tailgate=True
Frame Number=1375 Tailgate=True
Frame Number=1376 Tailgate=True
Frame Number=1377 Tailgate=True
Frame Number=1378 Tailgate=True
Frame Number=1379 Tailgate=True
Frame Number=1380 Tailgate=True
Frame Number=1381 Tailgate=True
Frame Number=1382 Tailgate=True
Frame Number=1383 Tailgate=True
Frame Number=1384 Tailgate=True
Frame Number=1385 Tailgate=True
Frame Number=1386 Tailgate=True
Frame Number=1387 Tailgate=True
Frame Number=1388 Tailgate=True
Frame Number=1389 Tailgate=True
Frame Number=1390 Tailgate=True
Frame Number=1391 Tailgate=True
Frame Number=1392 Tailgate=True
Frame Number=1393 Tailgate=True
Frame Number=1394 Tailgate=True
Frame Number=1395 Tailgate=True
Frame Number=1396 Tailgate=True
Frame Number=1397 Tailgate=True
Frame Number=1398 Tailgate=True
Frame Number=1399 Tailgate=True
Frame Number=1400 Tailgate=False
Frame Number=1401 Tailgate=False
Frame Number=1402 Tailgate=False
Frame Number=1403 Tailgate=False
Frame Number=1404 Tailgate=True
Frame Number=1405 Tailgate=True
Frame Number=1406 Tailgate=True
Frame Number=1407 Tailgate=True
Frame Number=1408 Tailgate=True
Frame Number=1409 Tailgate=True
Frame Number=1410 Tailgate=True
Frame Number=1411 Tailgate=True
Frame Number=1412 Tailgate=True
Frame Number=1413 Tailgate=True
Frame Number=1414 Tailgate=True
Frame Number=1415 Tailgate=True
Frame Number=1416 Tailgate=True
Frame Number=1417 Tailgate=True
Frame Number=1418 Tailgate=True
Frame Number=1419 Tailgate=True
Frame Number=1420 Tailgate=True
Frame Number=1421 Tailgate=True
Frame Number=1422 Tailgate=True
Frame Number=1423 Tailgate=True
Frame Number=1424 Tailgate=True
Frame Number=1425 Tailgate=True
Frame Number=1426 Tailgate=True
Frame Number=1427 Tailgate=True
Frame Number=1428 Tailgate=True
Frame Number=1429 Tailgate=True
Frame Number=1430 Tailgate=True
Frame Number=1431 Tailgate=True
Frame Number=1432 Tailgate=False
Frame Number=1433 Tailgate=False
Frame Number=1434 Tailgate=False
Frame Number=1435 Tailgate=False
Frame Number=1436 Tailgate=True
Frame Number=1437 Tailgate=True
Frame Number=1438 Tailgate=False
Frame Number=1439 Tailgate=False
Frame Number=1440 Tailgate=True
Frame Number=1441 Tailgate=False
Frame Number=1442 Tailgate=True
Frame Number=1443 Tailgate=True
Frame Number=1444 Tailgate=False
Frame Number=1445 Tailgate=False
Frame Number=1446 Tailgate=False
Frame Number=1447 Tailgate=False
Frame Number=1448 Tailgate=False
Frame Number=1449 Tailgate=False
Frame Number=1450 Tailgate=False
Frame Number=1451 Tailgate=False
Frame Number=1452 Tailgate=False
Frame Number=1453 Tailgate=False
Frame Number=1454 Tailgate=False
Frame Number=1455 Tailgate=False
Frame Number=1456 Tailgate=False
Frame Number=1457 Tailgate=False
Frame Number=1458 Tailgate=False
Frame Number=1459 Tailgate=False
Frame Number=1460 Tailgate=False
Frame Number=1461 Tailgate=False
Frame Number=1462 Tailgate=False
Frame Number=1463 Tailgate=False
Frame Number=1464 Tailgate=False
Frame Number=1465 Tailgate=False
Frame Number=1466 Tailgate=False
Frame Number=1467 Tailgate=False
Frame Number=1468 Tailgate=False
Frame Number=1469 Tailgate=False
Frame Number=1470 Tailgate=False
Frame Number=1471 Tailgate=False
Frame Number=1472 Tailgate=False
Frame Number=1473 Tailgate=False
Frame Number=1474 Tailgate=False
Frame Number=1475 Tailgate=False
Frame Number=1476 Tailgate=False
Frame Number=1477 Tailgate=False
Frame Number=1478 Tailgate=False
Frame Number=1479 Tailgate=False
Frame Number=1480 Tailgate=False
Frame Number=1481 Tailgate=False
Frame Number=1482 Tailgate=False
Frame Number=1483 Tailgate=False
Frame Number=1484 Tailgate=False
Frame Number=1485 Tailgate=False
Frame Number=1486 Tailgate=False
Frame Number=1487 Tailgate=False
Frame Number=1488 Tailgate=False
Frame Number=1489 Tailgate=False
Frame Number=1490 Tailgate=False
Frame Number=1491 Tailgate=False
Frame Number=1492 Tailgate=False
Frame Number=1493 Tailgate=False
Frame Number=1494 Tailgate=False
Frame Number=1495 Tailgate=False
Frame Number=1496 Tailgate=False
Frame Number=1497 Tailgate=False
Frame Number=1498 Tailgate=False
Frame Number=1499 Tailgate=False
Frame Number=1500 Tailgate=False
Frame Number=1501 Tailgate=False
Frame Number=1502 Tailgate=False
Frame Number=1503 Tailgate=False
Frame Number=1504 Tailgate=False
Frame Number=1505 Tailgate=False
Frame Number=1506 Tailgate=False
Frame Number=1507 Tailgate=False
Frame Number=1508 Tailgate=False
Frame Number=1509 Tailgate=False
Frame Number=1510 Tailgate=False
Frame Number=1511 Tailgate=False
Frame Number=1512 Tailgate=False
Frame Number=1513 Tailgate=False
Frame Number=1514 Tailgate=False
Frame Number=1515 Tailgate=False
Frame Number=1516 Tailgate=False
Frame Number=1517 Tailgate=False
Frame Number=1518 Tailgate=False
Frame Number=1519 Tailgate=False
Frame Number=1520 Tailgate=False
Frame Number=1521 Tailgate=False
Frame Number=1522 Tailgate=False
Frame Number=1523 Tailgate=False
Frame Number=1524 Tailgate=False
Frame Number=1525 Tailgate=False
Frame Number=1526 Tailgate=False
Frame Number=1527 Tailgate=False
Frame Number=1528 Tailgate=False
Frame Number=1529 Tailgate=False
Frame Number=1530 Tailgate=False
Frame Number=1531 Tailgate=False
Frame Number=1532 Tailgate=False
Frame Number=1533 Tailgate=False
Frame Number=1534 Tailgate=False
Frame Number=1535 Tailgate=False
Frame Number=1536 Tailgate=False
Frame Number=1537 Tailgate=False
Frame Number=1538 Tailgate=False
Frame Number=1539 Tailgate=False
Frame Number=1540 Tailgate=False
Frame Number=1541 Tailgate=False
Frame Number=1542 Tailgate=False
Frame Number=1543 Tailgate=False
Frame Number=1544 Tailgate=False
Frame Number=1545 Tailgate=False
Frame Number=1546 Tailgate=False
Frame Number=1547 Tailgate=False
Frame Number=1548 Tailgate=False
Frame Number=1549 Tailgate=False
Frame Number=1550 Tailgate=False
Frame Number=1551 Tailgate=False
Frame Number=1552 Tailgate=False
Frame Number=1553 Tailgate=False
Frame Number=1554 Tailgate=False
Frame Number=1555 Tailgate=False
Frame Number=1556 Tailgate=False
Frame Number=1557 Tailgate=False
Frame Number=1558 Tailgate=False
Frame Number=1559 Tailgate=False
Frame Number=1560 Tailgate=False
Frame Number=1561 Tailgate=False
Frame Number=1562 Tailgate=False
Frame Number=1563 Tailgate=False
Frame Number=1564 Tailgate=False
Frame Number=1565 Tailgate=False
Frame Number=1566 Tailgate=False
Frame Number=1567 Tailgate=False
Frame Number=1568 Tailgate=False
Frame Number=1569 Tailgate=False
Frame Number=1570 Tailgate=False
Frame Number=1571 Tailgate=False
Frame Number=1572 Tailgate=False
Frame Number=1573 Tailgate=False
Frame Number=1574 Tailgate=False
Frame Number=1575 Tailgate=False
Frame Number=1576 Tailgate=False
Frame Number=1577 Tailgate=False
Frame Number=1578 Tailgate=False
Frame Number=1579 Tailgate=False
Frame Number=1580 Tailgate=False
Frame Number=1581 Tailgate=False
Frame Number=1582 Tailgate=False
Frame Number=1583 Tailgate=False
Frame Number=1584 Tailgate=False
Frame Number=1585 Tailgate=False
Frame Number=1586 Tailgate=False
Frame Number=1587 Tailgate=False
Frame Number=1588 Tailgate=False
Frame Number=1589 Tailgate=False
Frame Number=1590 Tailgate=False
Frame Number=1591 Tailgate=False
Frame Number=1592 Tailgate=False
Frame Number=1593 Tailgate=False
Frame Number=1594 Tailgate=False
Frame Number=1595 Tailgate=False
Frame Number=1596 Tailgate=False
Frame Number=1597 Tailgate=False
Frame Number=1598 Tailgate=False
Frame Number=1599 Tailgate=False
Frame Number=1600 Tailgate=False
Frame Number=1601 Tailgate=False
Frame Number=1602 Tailgate=False
Frame Number=1603 Tailgate=False
Frame Number=1604 Tailgate=False
Frame Number=1605 Tailgate=False
Frame Number=1606 Tailgate=False
Frame Number=1607 Tailgate=False
Frame Number=1608 Tailgate=False
Frame Number=1609 Tailgate=False
Frame Number=1610 Tailgate=False
Frame Number=1611 Tailgate=False
Frame Number=1612 Tailgate=False
Frame Number=1613 Tailgate=False
Frame Number=1614 Tailgate=False
Frame Number=1615 Tailgate=False
Frame Number=1616 Tailgate=False
Frame Number=1617 Tailgate=False
Frame Number=1618 Tailgate=False
Frame Number=1619 Tailgate=False
Frame Number=1620 Tailgate=False
Frame Number=1621 Tailgate=False
Frame Number=1622 Tailgate=False
Frame Number=1623 Tailgate=False
Frame Number=1624 Tailgate=False
Frame Number=1625 Tailgate=False
Frame Number=1626 Tailgate=False
Frame Number=1627 Tailgate=False
Frame Number=1628 Tailgate=False
Frame Number=1629 Tailgate=False
Frame Number=1630 Tailgate=False
Frame Number=1631 Tailgate=False
Frame Number=1632 Tailgate=False
Frame Number=1633 Tailgate=False
Frame Number=1634 Tailgate=False
Frame Number=1635 Tailgate=False
Frame Number=1636 Tailgate=False
Frame Number=1637 Tailgate=False
Frame Number=1638 Tailgate=False
Frame Number=1639 Tailgate=False
Frame Number=1640 Tailgate=False
Frame Number=1641 Tailgate=False
Frame Number=1642 Tailgate=False
Frame Number=1643 Tailgate=False
Frame Number=1644 Tailgate=False
Frame Number=1645 Tailgate=False
Frame Number=1646 Tailgate=False
Frame Number=1647 Tailgate=False
Frame Number=1648 Tailgate=False
Frame Number=1649 Tailgate=False
Frame Number=1650 Tailgate=False
Frame Number=1651 Tailgate=False
Frame Number=1652 Tailgate=False
Frame Number=1653 Tailgate=False
Frame Number=1654 Tailgate=False
Frame Number=1655 Tailgate=False
Frame Number=1656 Tailgate=False
Frame Number=1657 Tailgate=False
Frame Number=1658 Tailgate=False
Frame Number=1659 Tailgate=False
Frame Number=1660 Tailgate=False
Frame Number=1661 Tailgate=False
Frame Number=1662 Tailgate=False
Frame Number=1663 Tailgate=False
Frame Number=1664 Tailgate=False
Frame Number=1665 Tailgate=False
Frame Number=1666 Tailgate=False
Frame Number=1667 Tailgate=False
Frame Number=1668 Tailgate=False
Frame Number=1669 Tailgate=False
Frame Number=1670 Tailgate=False
Frame Number=1671 Tailgate=False
Frame Number=1672 Tailgate=False
Frame Number=1673 Tailgate=False
Frame Number=1674 Tailgate=False
Frame Number=1675 Tailgate=False
Frame Number=1676 Tailgate=False
Frame Number=1677 Tailgate=False
Frame Number=1678 Tailgate=False
Frame Number=1679 Tailgate=False
Frame Number=1680 Tailgate=False
Frame Number=1681 Tailgate=False
Frame Number=1682 Tailgate=False
Frame Number=1683 Tailgate=False
Frame Number=1684 Tailgate=False
Frame Number=1685 Tailgate=False
Frame Number=1686 Tailgate=False
Frame Number=1687 Tailgate=False
Frame Number=1688 Tailgate=False
Frame Number=1689 Tailgate=False
Frame Number=1690 Tailgate=False
Frame Number=1691 Tailgate=False
Frame Number=1692 Tailgate=False
Frame Number=1693 Tailgate=False
Frame Number=1694 Tailgate=False
Frame Number=1695 Tailgate=False
Frame Number=1696 Tailgate=False
Frame Number=1697 Tailgate=False
Frame Number=1698 Tailgate=False
Frame Number=1699 Tailgate=False
Frame Number=1700 Tailgate=False
Frame Number=1701 Tailgate=False
Frame Number=1702 Tailgate=False
Frame Number=1703 Tailgate=False
Frame Number=1704 Tailgate=False
Frame Number=1705 Tailgate=False
Frame Number=1706 Tailgate=False
Frame Number=1707 Tailgate=False
Frame Number=1708 Tailgate=False
Frame Number=1709 Tailgate=False
Frame Number=1710 Tailgate=False
Frame Number=1711 Tailgate=False
Frame Number=1712 Tailgate=False
Frame Number=1713 Tailgate=False
Frame Number=1714 Tailgate=False
Frame Number=1715 Tailgate=False
Frame Number=1716 Tailgate=False
Frame Number=1717 Tailgate=False
Frame Number=1718 Tailgate=False
Frame Number=1719 Tailgate=False
Frame Number=1720 Tailgate=False
Frame Number=1721 Tailgate=False
Frame Number=1722 Tailgate=False
Frame Number=1723 Tailgate=False
Frame Number=1724 Tailgate=False
Frame Number=1725 Tailgate=False
Frame Number=1726 Tailgate=False
Frame Number=1727 Tailgate=False
Frame Number=1728 Tailgate=False
Frame Number=1729 Tailgate=False
Frame Number=1730 Tailgate=False
Frame Number=1731 Tailgate=False
Frame Number=1732 Tailgate=False
Frame Number=1733 Tailgate=False
Frame Number=1734 Tailgate=False
Frame Number=1735 Tailgate=False
Frame Number=1736 Tailgate=False
Frame Number=1737 Tailgate=False
Frame Number=1738 Tailgate=False
Frame Number=1739 Tailgate=False
Frame Number=1740 Tailgate=False
Frame Number=1741 Tailgate=False
Frame Number=1742 Tailgate=False
Frame Number=1743 Tailgate=False
Frame Number=1744 Tailgate=False
Frame Number=1745 Tailgate=False
Frame Number=1746 Tailgate=False
Frame Number=1747 Tailgate=False
Frame Number=1748 Tailgate=False
Frame Number=1749 Tailgate=False
Frame Number=1750 Tailgate=False
Frame Number=1751 Tailgate=False
Frame Number=1752 Tailgate=False
Frame Number=1753 Tailgate=False
Frame Number=1754 Tailgate=False
Frame Number=1755 Tailgate=False
Frame Number=1756 Tailgate=False
Frame Number=1757 Tailgate=False
Frame Number=1758 Tailgate=False
Frame Number=1759 Tailgate=False
Frame Number=1760 Tailgate=False
Frame Number=1761 Tailgate=False
Frame Number=1762 Tailgate=False
Frame Number=1763 Tailgate=False
Frame Number=1764 Tailgate=False
Frame Number=1765 Tailgate=False
Frame Number=1766 Tailgate=False
Frame Number=1767 Tailgate=False
Frame Number=1768 Tailgate=False
Frame Number=1769 Tailgate=False
Frame Number=1770 Tailgate=False
Frame Number=1771 Tailgate=False
Frame Number=1772 Tailgate=False
Frame Number=1773 Tailgate=False
Frame Number=1774 Tailgate=False
Frame Number=1775 Tailgate=False
Frame Number=1776 Tailgate=False
Frame Number=1777 Tailgate=False
Frame Number=1778 Tailgate=False
Frame Number=1779 Tailgate=False
Frame Number=1780 Tailgate=False
Frame Number=1781 Tailgate=False
Frame Number=1782 Tailgate=False
Frame Number=1783 Tailgate=False
Frame Number=1784 Tailgate=False
Frame Number=1785 Tailgate=False
Frame Number=1786 Tailgate=False
Frame Number=1787 Tailgate=False
Frame Number=1788 Tailgate=False
Frame Number=1789 Tailgate=False
Frame Number=1790 Tailgate=False
Frame Number=1791 Tailgate=False
Frame Number=1792 Tailgate=False
Frame Number=1793 Tailgate=False
Frame Number=1794 Tailgate=False
Frame Number=1795 Tailgate=False
Frame Number=1796 Tailgate=False
Frame Number=1797 Tailgate=False
Frame Number=1798 Tailgate=False
Frame Number=1799 Tailgate=False
Frame Number=1800 Tailgate=False
Frame Number=1801 Tailgate=False
Frame Number=1802 Tailgate=False
Frame Number=1803 Tailgate=False
Frame Number=1804 Tailgate=False
Frame Number=1805 Tailgate=False
Frame Number=1806 Tailgate=False
Frame Number=1807 Tailgate=False
Frame Number=1808 Tailgate=False
Frame Number=1809 Tailgate=False
Frame Number=1810 Tailgate=False
Frame Number=1811 Tailgate=False
Frame Number=1812 Tailgate=False
Frame Number=1813 Tailgate=False
Frame Number=1814 Tailgate=False
Frame Number=1815 Tailgate=False
Frame Number=1816 Tailgate=False
Frame Number=1817 Tailgate=False
Frame Number=1818 Tailgate=False
Frame Number=1819 Tailgate=False
Frame Number=1820 Tailgate=False
Frame Number=1821 Tailgate=False
Frame Number=1822 Tailgate=False
Frame Number=1823 Tailgate=False
Frame Number=1824 Tailgate=False
Frame Number=1825 Tailgate=False
Frame Number=1826 Tailgate=False
Frame Number=1827 Tailgate=False
Frame Number=1828 Tailgate=False
Frame Number=1829 Tailgate=False
Frame Number=1830 Tailgate=False
Frame Number=1831 Tailgate=False
Frame Number=1832 Tailgate=False
Frame Number=1833 Tailgate=False
Frame Number=1834 Tailgate=False
Frame Number=1835 Tailgate=False
Frame Number=1836 Tailgate=False
Frame Number=1837 Tailgate=False
Frame Number=1838 Tailgate=False
Frame Number=1839 Tailgate=False
Frame Number=1840 Tailgate=False
Frame Number=1841 Tailgate=False
Frame Number=1842 Tailgate=False
Frame Number=1843 Tailgate=False
Frame Number=1844 Tailgate=False
Frame Number=1845 Tailgate=False
Frame Number=1846 Tailgate=False
Frame Number=1847 Tailgate=False
Frame Number=1848 Tailgate=False
Frame Number=1849 Tailgate=False
Frame Number=1850 Tailgate=False
Frame Number=1851 Tailgate=False
Frame Number=1852 Tailgate=False
Frame Number=1853 Tailgate=False
Frame Number=1854 Tailgate=False
Frame Number=1855 Tailgate=False
Frame Number=1856 Tailgate=False
Frame Number=1857 Tailgate=False
Frame Number=1858 Tailgate=False
Frame Number=1859 Tailgate=False
Frame Number=1860 Tailgate=False
Frame Number=1861 Tailgate=False
Frame Number=1862 Tailgate=False
Frame Number=1863 Tailgate=False
Frame Number=1864 Tailgate=False
Frame Number=1865 Tailgate=False
Frame Number=1866 Tailgate=False
Frame Number=1867 Tailgate=False
Frame Number=1868 Tailgate=False
Frame Number=1869 Tailgate=False
Frame Number=1870 Tailgate=False
Frame Number=1871 Tailgate=False
Frame Number=1872 Tailgate=False
Frame Number=1873 Tailgate=False
Frame Number=1874 Tailgate=False
Frame Number=1875 Tailgate=False
Frame Number=1876 Tailgate=False
Frame Number=1877 Tailgate=False
Frame Number=1878 Tailgate=False
Frame Number=1879 Tailgate=False
Frame Number=1880 Tailgate=False
Frame Number=1881 Tailgate=False
Frame Number=1882 Tailgate=False
Frame Number=1883 Tailgate=False
Frame Number=1884 Tailgate=False
Frame Number=1885 Tailgate=False
Frame Number=1886 Tailgate=False
Frame Number=1887 Tailgate=False
Frame Number=1888 Tailgate=False
Frame Number=1889 Tailgate=False
Frame Number=1890 Tailgate=False
Frame Number=1891 Tailgate=False
Frame Number=1892 Tailgate=False
Frame Number=1893 Tailgate=False
Frame Number=1894 Tailgate=False
Frame Number=1895 Tailgate=False
Frame Number=1896 Tailgate=False
Frame Number=1897 Tailgate=False
Frame Number=1898 Tailgate=False
Frame Number=1899 Tailgate=False
Frame Number=1900 Tailgate=False
Frame Number=1901 Tailgate=False
Frame Number=1902 Tailgate=False
Frame Number=1903 Tailgate=False
Frame Number=1904 Tailgate=False
Frame Number=1905 Tailgate=False
Frame Number=1906 Tailgate=False
Frame Number=1907 Tailgate=False
Frame Number=1908 Tailgate=False
Frame Number=1909 Tailgate=False
Frame Number=1910 Tailgate=False
Frame Number=1911 Tailgate=False
Frame Number=1912 Tailgate=False
Frame Number=1913 Tailgate=False
Frame Number=1914 Tailgate=False
Frame Number=1915 Tailgate=False
Frame Number=1916 Tailgate=False
Frame Number=1917 Tailgate=False
Frame Number=1918 Tailgate=False
Frame Number=1919 Tailgate=False
Frame Number=1920 Tailgate=False
Frame Number=1921 Tailgate=False
Frame Number=1922 Tailgate=False
Frame Number=1923 Tailgate=False
Frame Number=1924 Tailgate=False
Frame Number=1925 Tailgate=False
Frame Number=1926 Tailgate=False
Frame Number=1927 Tailgate=False
Frame Number=1928 Tailgate=False
Frame Number=1929 Tailgate=False
Frame Number=1930 Tailgate=False
Frame Number=1931 Tailgate=False
Frame Number=1932 Tailgate=False
Frame Number=1933 Tailgate=False
Frame Number=1934 Tailgate=False
Frame Number=1935 Tailgate=False
Frame Number=1936 Tailgate=False
Frame Number=1937 Tailgate=False
Frame Number=1938 Tailgate=False
Frame Number=1939 Tailgate=False
Frame Number=1940 Tailgate=False
Frame Number=1941 Tailgate=False
Frame Number=1942 Tailgate=False
Frame Number=1943 Tailgate=False
Frame Number=1944 Tailgate=False
Frame Number=1945 Tailgate=False
Frame Number=1946 Tailgate=False
Frame Number=1947 Tailgate=False
Frame Number=1948 Tailgate=False
Frame Number=1949 Tailgate=False
Frame Number=1950 Tailgate=False
Frame Number=1951 Tailgate=False
Frame Number=1952 Tailgate=False
Frame Number=1953 Tailgate=False
Frame Number=1954 Tailgate=False
Frame Number=1955 Tailgate=False
Frame Number=1956 Tailgate=False
Frame Number=1957 Tailgate=False
Frame Number=1958 Tailgate=False
Frame Number=1959 Tailgate=False
Frame Number=1960 Tailgate=False
Frame Number=1961 Tailgate=False
Frame Number=1962 Tailgate=False
Frame Number=1963 Tailgate=False
Frame Number=1964 Tailgate=False
Frame Number=1965 Tailgate=False
Frame Number=1966 Tailgate=False
Frame Number=1967 Tailgate=False
Frame Number=1968 Tailgate=False
Frame Number=1969 Tailgate=False
Frame Number=1970 Tailgate=False
Frame Number=1971 Tailgate=False
Frame Number=1972 Tailgate=False
Frame Number=1973 Tailgate=False
Frame Number=1974 Tailgate=False
Frame Number=1975 Tailgate=False
Frame Number=1976 Tailgate=False
Frame Number=1977 Tailgate=False
Frame Number=1978 Tailgate=False
Frame Number=1979 Tailgate=False
Frame Number=1980 Tailgate=False
Frame Number=1981 Tailgate=False
Frame Number=1982 Tailgate=False
Frame Number=1983 Tailgate=False
Frame Number=1984 Tailgate=False
Frame Number=1985 Tailgate=False
Frame Number=1986 Tailgate=False
Frame Number=1987 Tailgate=False
Frame Number=1988 Tailgate=False
Frame Number=1989 Tailgate=False
Frame Number=1990 Tailgate=False
Frame Number=1991 Tailgate=False
Frame Number=1992 Tailgate=False
Frame Number=1993 Tailgate=False
Frame Number=1994 Tailgate=False
Frame Number=1995 Tailgate=False
Frame Number=1996 Tailgate=False
Frame Number=1997 Tailgate=False
Frame Number=1998 Tailgate=False
Frame Number=1999 Tailgate=False
Frame Number=2000 Tailgate=False
Frame Number=2001 Tailgate=False
Frame Number=2002 Tailgate=False
Frame Number=2003 Tailgate=False
Frame Number=2004 Tailgate=False
Frame Number=2005 Tailgate=False
Frame Number=2006 Tailgate=False
Frame Number=2007 Tailgate=False
Frame Number=2008 Tailgate=False
Frame Number=2009 Tailgate=False
Frame Number=2010 Tailgate=False
Frame Number=2011 Tailgate=False
Frame Number=2012 Tailgate=False
Frame Number=2013 Tailgate=False
Frame Number=2014 Tailgate=False
Frame Number=2015 Tailgate=False
Frame Number=2016 Tailgate=False
Frame Number=2017 Tailgate=False
Frame Number=2018 Tailgate=False
Frame Number=2019 Tailgate=False
Frame Number=2020 Tailgate=False
Frame Number=2021 Tailgate=False
Frame Number=2022 Tailgate=False
Frame Number=2023 Tailgate=False
Frame Number=2024 Tailgate=False
Frame Number=2025 Tailgate=False
Frame Number=2026 Tailgate=False
Frame Number=2027 Tailgate=False
Frame Number=2028 Tailgate=False
Frame Number=2029 Tailgate=False
Frame Number=2030 Tailgate=False
Frame Number=2031 Tailgate=False
Frame Number=2032 Tailgate=False
Frame Number=2033 Tailgate=False
Frame Number=2034 Tailgate=False
Frame Number=2035 Tailgate=False
Frame Number=2036 Tailgate=False
Frame Number=2037 Tailgate=False
Frame Number=2038 Tailgate=False
Frame Number=2039 Tailgate=False
Frame Number=2040 Tailgate=False
Frame Number=2041 Tailgate=False
Frame Number=2042 Tailgate=False
Frame Number=2043 Tailgate=False
Frame Number=2044 Tailgate=False
Frame Number=2045 Tailgate=False
Frame Number=2046 Tailgate=False
Frame Number=2047 Tailgate=False
Frame Number=2048 Tailgate=False
Frame Number=2049 Tailgate=False
Frame Number=2050 Tailgate=False
Frame Number=2051 Tailgate=False
Frame Number=2052 Tailgate=True
Frame Number=2053 Tailgate=True
Frame Number=2054 Tailgate=False
Frame Number=2055 Tailgate=False
Frame Number=2056 Tailgate=False
Frame Number=2057 Tailgate=False
Frame Number=2058 Tailgate=False
Frame Number=2059 Tailgate=False
Frame Number=2060 Tailgate=False
Frame Number=2061 Tailgate=False
Frame Number=2062 Tailgate=False
Frame Number=2063 Tailgate=False
Frame Number=2064 Tailgate=False
Frame Number=2065 Tailgate=False
Frame Number=2066 Tailgate=False
Frame Number=2067 Tailgate=False
Frame Number=2068 Tailgate=False
Frame Number=2069 Tailgate=False
Frame Number=2070 Tailgate=False
Frame Number=2071 Tailgate=False
Frame Number=2072 Tailgate=False
Frame Number=2073 Tailgate=False
Frame Number=2074 Tailgate=False
Frame Number=2075 Tailgate=False
Frame Number=2076 Tailgate=False
Frame Number=2077 Tailgate=False
Frame Number=2078 Tailgate=False
Frame Number=2079 Tailgate=False
Frame Number=2080 Tailgate=False
Frame Number=2081 Tailgate=False
Frame Number=2082 Tailgate=False
Frame Number=2083 Tailgate=False
Frame Number=2084 Tailgate=False
Frame Number=2085 Tailgate=False
Frame Number=2086 Tailgate=False
Frame Number=2087 Tailgate=False
Frame Number=2088 Tailgate=False
Frame Number=2089 Tailgate=False
Frame Number=2090 Tailgate=False
Frame Number=2091 Tailgate=False
Frame Number=2092 Tailgate=False
Frame Number=2093 Tailgate=False
Frame Number=2094 Tailgate=False
Frame Number=2095 Tailgate=False
Frame Number=2096 Tailgate=False
Frame Number=2097 Tailgate=False
Frame Number=2098 Tailgate=False
Frame Number=2099 Tailgate=False
Frame Number=2100 Tailgate=False
Frame Number=2101 Tailgate=False
Frame Number=2102 Tailgate=False
Frame Number=2103 Tailgate=False
Frame Number=2104 Tailgate=False
Frame Number=2105 Tailgate=False
Frame Number=2106 Tailgate=False
Frame Number=2107 Tailgate=False
Frame Number=2108 Tailgate=False
Frame Number=2109 Tailgate=False
Frame Number=2110 Tailgate=False
Frame Number=2111 Tailgate=False
Frame Number=2112 Tailgate=False
Frame Number=2113 Tailgate=False
Frame Number=2114 Tailgate=False
Frame Number=2115 Tailgate=False
Frame Number=2116 Tailgate=False
Frame Number=2117 Tailgate=False
Frame Number=2118 Tailgate=False
Frame Number=2119 Tailgate=False
Frame Number=2120 Tailgate=False
Frame Number=2121 Tailgate=False
Frame Number=2122 Tailgate=False
Frame Number=2123 Tailgate=False
Frame Number=2124 Tailgate=False
Frame Number=2125 Tailgate=False
Frame Number=2126 Tailgate=False
Frame Number=2127 Tailgate=False
Frame Number=2128 Tailgate=False
Frame Number=2129 Tailgate=False
Frame Number=2130 Tailgate=False
Frame Number=2131 Tailgate=False
Frame Number=2132 Tailgate=False
Frame Number=2133 Tailgate=False
Frame Number=2134 Tailgate=False
Frame Number=2135 Tailgate=False
Frame Number=2136 Tailgate=False
Frame Number=2137 Tailgate=False
Frame Number=2138 Tailgate=False
Frame Number=2139 Tailgate=False
Frame Number=2140 Tailgate=False
Frame Number=2141 Tailgate=False
Frame Number=2142 Tailgate=False
Frame Number=2143 Tailgate=False
Frame Number=2144 Tailgate=False
Frame Number=2145 Tailgate=False
Frame Number=2146 Tailgate=False
Frame Number=2147 Tailgate=False
Frame Number=2148 Tailgate=False
Frame Number=2149 Tailgate=False
Frame Number=2150 Tailgate=False
Frame Number=2151 Tailgate=False
Frame Number=2152 Tailgate=False
Frame Number=2153 Tailgate=False
Frame Number=2154 Tailgate=False
Frame Number=2155 Tailgate=False
Frame Number=2156 Tailgate=False
Frame Number=2157 Tailgate=False
Frame Number=2158 Tailgate=False
Frame Number=2159 Tailgate=False
Frame Number=2160 Tailgate=False
Frame Number=2161 Tailgate=False
Frame Number=2162 Tailgate=False
Frame Number=2163 Tailgate=False
Frame Number=2164 Tailgate=False
Frame Number=2165 Tailgate=False
Frame Number=2166 Tailgate=False
Frame Number=2167 Tailgate=False
Frame Number=2168 Tailgate=False
Frame Number=2169 Tailgate=False
Frame Number=2170 Tailgate=False
Frame Number=2171 Tailgate=False
Frame Number=2172 Tailgate=False
Frame Number=2173 Tailgate=False
Frame Number=2174 Tailgate=False
Frame Number=2175 Tailgate=False
Frame Number=2176 Tailgate=False
Frame Number=2177 Tailgate=False
Frame Number=2178 Tailgate=False
Frame Number=2179 Tailgate=False
Frame Number=2180 Tailgate=False
Frame Number=2181 Tailgate=False
Frame Number=2182 Tailgate=False
Frame Number=2183 Tailgate=False
Frame Number=2184 Tailgate=False
Frame Number=2185 Tailgate=False
Frame Number=2186 Tailgate=False
Frame Number=2187 Tailgate=False
Frame Number=2188 Tailgate=False
Frame Number=2189 Tailgate=False
Frame Number=2190 Tailgate=False
Frame Number=2191 Tailgate=False
Frame Number=2192 Tailgate=False
Frame Number=2193 Tailgate=False
Frame Number=2194 Tailgate=False
Frame Number=2195 Tailgate=False
Frame Number=2196 Tailgate=False
Frame Number=2197 Tailgate=False
Frame Number=2198 Tailgate=False
Frame Number=2199 Tailgate=False
Frame Number=2200 Tailgate=False
Frame Number=2201 Tailgate=False
Frame Number=2202 Tailgate=False
Frame Number=2203 Tailgate=False
Frame Number=2204 Tailgate=False
Frame Number=2205 Tailgate=False
Frame Number=2206 Tailgate=False
Frame Number=2207 Tailgate=False
Frame Number=2208 Tailgate=False
Frame Number=2209 Tailgate=False
Frame Number=2210 Tailgate=False
Frame Number=2211 Tailgate=False
Frame Number=2212 Tailgate=False
Frame Number=2213 Tailgate=False
Frame Number=2214 Tailgate=False
Frame Number=2215 Tailgate=False
Frame Number=2216 Tailgate=False
Frame Number=2217 Tailgate=False
Frame Number=2218 Tailgate=False
Frame Number=2219 Tailgate=False
Frame Number=2220 Tailgate=False
Frame Number=2221 Tailgate=False
Frame Number=2222 Tailgate=False
Frame Number=2223 Tailgate=False
Frame Number=2224 Tailgate=False
Frame Number=2225 Tailgate=False
Frame Number=2226 Tailgate=False
Frame Number=2227 Tailgate=False
Frame Number=2228 Tailgate=False
Frame Number=2229 Tailgate=False
Frame Number=2230 Tailgate=False
Frame Number=2231 Tailgate=False
Frame Number=2232 Tailgate=False
Frame Number=2233 Tailgate=False
Frame Number=2234 Tailgate=False
Frame Number=2235 Tailgate=False
Frame Number=2236 Tailgate=False
Frame Number=2237 Tailgate=False
Frame Number=2238 Tailgate=False
Frame Number=2239 Tailgate=False
Frame Number=2240 Tailgate=False
Frame Number=2241 Tailgate=False
Frame Number=2242 Tailgate=False
Frame Number=2243 Tailgate=False
Frame Number=2244 Tailgate=False
Frame Number=2245 Tailgate=False
Frame Number=2246 Tailgate=False
Frame Number=2247 Tailgate=False
Frame Number=2248 Tailgate=False
Frame Number=2249 Tailgate=False
Frame Number=2250 Tailgate=False
Frame Number=2251 Tailgate=False
Frame Number=2252 Tailgate=False
Frame Number=2253 Tailgate=False
Frame Number=2254 Tailgate=False
Frame Number=2255 Tailgate=False
Frame Number=2256 Tailgate=False
Frame Number=2257 Tailgate=False
Frame Number=2258 Tailgate=False
Frame Number=2259 Tailgate=False
Frame Number=2260 Tailgate=False
Frame Number=2261 Tailgate=False
Frame Number=2262 Tailgate=False
Frame Number=2263 Tailgate=False
Frame Number=2264 Tailgate=False
Frame Number=2265 Tailgate=False
Frame Number=2266 Tailgate=False
Frame Number=2267 Tailgate=False
Frame Number=2268 Tailgate=False
Frame Number=2269 Tailgate=False
Frame Number=2270 Tailgate=False
Frame Number=2271 Tailgate=False
Frame Number=2272 Tailgate=False
Frame Number=2273 Tailgate=False
Frame Number=2274 Tailgate=False
Frame Number=2275 Tailgate=False
Frame Number=2276 Tailgate=False
Frame Number=2277 Tailgate=False
Frame Number=2278 Tailgate=False
Frame Number=2279 Tailgate=False
Frame Number=2280 Tailgate=False
Frame Number=2281 Tailgate=False
Frame Number=2282 Tailgate=False
Frame Number=2283 Tailgate=False
Frame Number=2284 Tailgate=False
Frame Number=2285 Tailgate=False
Frame Number=2286 Tailgate=False
Frame Number=2287 Tailgate=False
Frame Number=2288 Tailgate=False
Frame Number=2289 Tailgate=False
Frame Number=2290 Tailgate=False
Frame Number=2291 Tailgate=False
Frame Number=2292 Tailgate=False
Frame Number=2293 Tailgate=False
Frame Number=2294 Tailgate=False
Frame Number=2295 Tailgate=False
Frame Number=2296 Tailgate=False
Frame Number=2297 Tailgate=False
Frame Number=2298 Tailgate=False
Frame Number=2299 Tailgate=False
Frame Number=2300 Tailgate=False
Frame Number=2301 Tailgate=False
Frame Number=2302 Tailgate=False
Frame Number=2303 Tailgate=False
Frame Number=2304 Tailgate=False
Frame Number=2305 Tailgate=False
Frame Number=2306 Tailgate=False
Frame Number=2307 Tailgate=False
Frame Number=2308 Tailgate=False
Frame Number=2309 Tailgate=False
Frame Number=2310 Tailgate=False
Frame Number=2311 Tailgate=False
Frame Number=2312 Tailgate=False
Frame Number=2313 Tailgate=False
Frame Number=2314 Tailgate=False
Frame Number=2315 Tailgate=False
Frame Number=2316 Tailgate=False
Frame Number=2317 Tailgate=False
Frame Number=2318 Tailgate=False
Frame Number=2319 Tailgate=False
Frame Number=2320 Tailgate=False
Frame Number=2321 Tailgate=False
Frame Number=2322 Tailgate=False
Frame Number=2323 Tailgate=False
Frame Number=2324 Tailgate=False
Frame Number=2325 Tailgate=False
Frame Number=2326 Tailgate=False
Frame Number=2327 Tailgate=False
Frame Number=2328 Tailgate=False
Frame Number=2329 Tailgate=False
Frame Number=2330 Tailgate=False
Frame Number=2331 Tailgate=False
Frame Number=2332 Tailgate=False
Frame Number=2333 Tailgate=False
Frame Number=2334 Tailgate=False
Frame Number=2335 Tailgate=False
Frame Number=2336 Tailgate=False
Frame Number=2337 Tailgate=False
Frame Number=2338 Tailgate=False
Frame Number=2339 Tailgate=False
Frame Number=2340 Tailgate=False
Frame Number=2341 Tailgate=False
Frame Number=2342 Tailgate=False
Frame Number=2343 Tailgate=False
Frame Number=2344 Tailgate=False
Frame Number=2345 Tailgate=False
Frame Number=2346 Tailgate=False
Frame Number=2347 Tailgate=False
Frame Number=2348 Tailgate=False
Frame Number=2349 Tailgate=False
Frame Number=2350 Tailgate=False
Frame Number=2351 Tailgate=False
Frame Number=2352 Tailgate=False
Frame Number=2353 Tailgate=False
Frame Number=2354 Tailgate=False
Frame Number=2355 Tailgate=False
Frame Number=2356 Tailgate=False
Frame Number=2357 Tailgate=False
Frame Number=2358 Tailgate=False
Frame Number=2359 Tailgate=False
Frame Number=2360 Tailgate=False
Frame Number=2361 Tailgate=False
Frame Number=2362 Tailgate=False
Frame Number=2363 Tailgate=False
Frame Number=2364 Tailgate=False
Frame Number=2365 Tailgate=False
Frame Number=2366 Tailgate=False
Frame Number=2367 Tailgate=False
Frame Number=2368 Tailgate=False
Frame Number=2369 Tailgate=False
Frame Number=2370 Tailgate=False
Frame Number=2371 Tailgate=False
Frame Number=2372 Tailgate=False
Frame Number=2373 Tailgate=False
Frame Number=2374 Tailgate=False
Frame Number=2375 Tailgate=False
Frame Number=2376 Tailgate=False
Frame Number=2377 Tailgate=False
Frame Number=2378 Tailgate=False
Frame Number=2379 Tailgate=False
Frame Number=2380 Tailgate=False
Frame Number=2381 Tailgate=False
Frame Number=2382 Tailgate=False
Frame Number=2383 Tailgate=False
Frame Number=2384 Tailgate=False
Frame Number=2385 Tailgate=False
Frame Number=2386 Tailgate=False
Frame Number=2387 Tailgate=False
Frame Number=2388 Tailgate=False
Frame Number=2389 Tailgate=False
Frame Number=2390 Tailgate=False
Frame Number=2391 Tailgate=False
Frame Number=2392 Tailgate=False
Frame Number=2393 Tailgate=False
Frame Number=2394 Tailgate=False
Frame Number=2395 Tailgate=False
Frame Number=2396 Tailgate=False
Frame Number=2397 Tailgate=False
Frame Number=2398 Tailgate=False
Frame Number=2399 Tailgate=False
Frame Number=2400 Tailgate=False
Frame Number=2401 Tailgate=False
Frame Number=2402 Tailgate=False
Frame Number=2403 Tailgate=False
Frame Number=2404 Tailgate=False
Frame Number=2405 Tailgate=False
End-of-streamme 2405
## Step 5: Analyze the Results ##
Finally, we can analyze the driving behavior using the log we've collected. 
**Instructions**: <br>
5.1. Execute the cell to import the tailgate log into a Pandas DataFrame. <br>
5.2. Execute the cell to plot the occurences of tailgating. <br>
5.3. Make sure the output `.mp4` file is being referenced and execute the cell to view the composite with the bounding boxes drawn into the original video. <br>
5.4. Execute the cell to calculate the amount of time on average this vehicle spent tailgating. <br>
5.5. Modify the `<FIXME>` _only_ to mark your answer. <br>
# 5.1
# DO NOT CHANGE THIS CELL
import pandas as pd
​
df=pd.read_csv('my_assessment/answer_4.txt', names=['inference'])
df.head()
inference
0	0
1	0
2	0
3	0
4	0
# 5.2
# DO NOT CHANGE THIS CELL
import matplotlib.pyplot as plt
import numpy as np
​
df.plot(kind='bar', figsize=(30, 5))
plt.xticks(np.arange(0, len(df)+1, FRAME_RATE), np.arange(0, len(df)/FRAME_RATE))
plt.show()

# 5.3
# DO NOT CHANGE THIS CELL
!ffmpeg -i output.mpeg4 output_converted.mp4 \
        -y \
        -loglevel quiet
​
Video('output_converted.mp4', width=720)
# 5.4
# DO NOT CHANGE THIS CELL
display(df['inference'].value_counts(normalize=True))
0    0.881131
1    0.118869
Name: inference, dtype: float64
11.2454
# Question: How much time (without the percentage sign, e.g. 5.0) did the vehicle tailgate? 
Answer=11.2454
​
# EXAMPLE: 
# Answer='5.0'
​
# DO NOT CHANGE BELOW
!echo $Answer > my_assessment/answer_5.txt
## Grade Your Code ##
If you have completed all 5 questions and confirmed the pipeline runs correctly, save changes to the notebook and revisit the webpage where you launched this interactive environment. Click on the "**ASSESS TASK**" button as shown in the screenshot below. Doing so will give you credit for this part of the lab that counts towards earning a certificate of competency for the entire course. 
​
<p><img src='images/credit.png' width=1080></p>
### BONUS. Visualizing Frames ###
Below we have included some helpful functions that will help you visualize the frames that exhibit tailgating behavior. 
**Instructions**: <br>
B.1. Execute the cell to extract tailgating frames. <br>
B.2. Execute the cell to display randomly selected tailgating frames. <br>
# B.1
# DO NOT CHANGE THIS CELL
import cv2
​
!mkdir output_images
!rm -r output_images/*
input_video=cv2.VideoCapture('output_converted.mp4')
retVal, im=input_video.read()
frameCount=0
while retVal:
    if frameCount in df[df['inference']==1].index:
        cv2.imwrite("output_images/frame_%d.jpg" % frameCount, im)     # save frame as JPEG file      
    retVal, im=input_video.read()
    print(f'Read a new frame: {frameCount}', end='\r')
    frameCount+=1
input_video.release()
rm: cannot remove 'output_images/*': No such file or directory
Read a new frame: 2405
# B.2
# DO NOT CHANGE THIS CELL
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from matplotlib.pyplot import imshow
import numpy as np
​
def plot_random_samples(frames):
    sample_frames = np.random.choice(frames,size=8)
    fig=plt.figure(figsize=(30, 8))
    columns = 4
    rows = 2
    i = 1 
    for frame_num in sample_frames:
        # im = Image.open('{}/images/{}/{}.jpg'.format(config["Base_Dest_Folder"], config["Test_Video_ID"], box["frame_no"]))
        im = Image.open(f'output_images/frame_{frame_num}.jpg')
        fig.add_subplot(rows, columns, i)
        i += 1
        plt.imshow(np.asarray(im))
    plt.show()
    
plot_random_samples(df[df['inference']==1].index)

<a href="https://www.nvidia.com/dli"><img src="images/DLI_Header.png" alt="Header" style="width: 400px;"/></a>

Simple
1
2
Python 3 | Disconnected
assessment.ipynb
Ln 1, Col 1
Mode: Edit