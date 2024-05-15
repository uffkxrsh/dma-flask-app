import { EffectComposer } from "https://unpkg.com/three@0.120.0/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "https://unpkg.com/three@0.120.0/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "https://unpkg.com/three@0.120.0/examples/jsm/postprocessing/UnrealBloomPass.js";
import { OrbitControls } from "https://unpkg.com/three@0.120.0/examples/jsm/controls/OrbitControls";


var options = {
  exposure: 2.8,
  bloomStrength: 1.9,
  bloomThreshold: 0,
  bloomRadius: 0.55,
  color0: [74, 30, 0],
  color1: [201, 158, 72],
};
// var gui = new dat.GUI();

// var bloom = gui.addFolder("Bloom");
// // bloom.add(options, "exposure", 0.0, 5.0).name("exposure").listen();
// bloom.add(options, "bloomStrength", 0.0, 5.0).name("bloomStrength").listen();
// // bloom.add(options, "bloomThreshold", 0.0, 1.0).name("bloomThreshold").listen();
// bloom.add(options, "bloomRadius", 0.1, 2.0).name("bloomRadius").listen();
// bloom.open();

// var color = gui.addFolder("Colors");
// color.addColor(options, "color0").name("Border");
// color.addColor(options, "color1").name("Base");
// color.open();

const vert = `
      varying vec3 vNormal;
      varying vec3 camPos;
      varying vec3 vPosition;
      varying vec2 vUv;
      varying vec3 eyeVector;
      attribute vec3 center;
      varying vec3 vCenter;

      
      void main() {
        vNormal = normal;
        vCenter = center;
        camPos = cameraPosition;
        vPosition = position;
        vUv= uv;
        vec4 worldPosition = modelViewMatrix * vec4( position, 1.0);
        eyeVector = normalize(worldPosition.xyz - cameraPosition);
        gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
      }

`;

const frag = `

      #ifdef GL_ES
      precision lowp float;
      #endif

      #define NUM_OCTAVES 5
      uniform vec4 resolution;
      uniform vec3 color1;
      uniform vec3 color0;
      uniform float utime;
      uniform sampler2D colorRamp;
      uniform sampler2D noiseTex;
      varying vec3 camPos;
      varying vec3 vNormal;
      varying vec3 vPosition;
      varying vec2 vUv;
      varying vec3 eyeVector;
      varying vec3 vCenter; 

      float rand(vec2 n) {
        return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
      }

      float noise(vec2 p){
        vec2 ip = floor(p);
        vec2 u = fract(p);
        u = u*u*(3.0-2.0*u);

        float res = mix(
          mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
          mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
        return res*res;
      }

      float fbm(vec2 x) {
        float v = 0.0;
        float a = 0.5;
        vec2 shift = vec2(100);
        // Rotate to reduce axial bias
          mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
        for (int i = 0; i < NUM_OCTAVES; ++i) {
          v += a * noise(x);
          x = rot * x * 2.0 + shift;
          a *= 0.5;
        }
        return v;
      }

      vec3 rgbcol(float r, float g, float b) {
        return vec3(r/255.0,g/255.0,b/255.0);
      }

      float setOpacity(float r, float g, float b) {
        float tone = (r + g + b) / 3.0;
        float alpha = 1.0;
        if(tone<0.99) {
          alpha = 0.0;
        }
        return alpha;
      }

      float Fresnel(vec3 eyeVector, vec3 worldNormal) {
        return pow( 1.0 + dot( eyeVector, worldNormal), 3.0 );
      }


      float getTone(vec3 color){
        float tone = (color.r + color.g + color.b) / 3.0;
        return tone;
      }

      float edgeFactorTri() {
        vec3 d = fwidth(vCenter.xyz);
        vec3 a3 = smoothstep(vec3(0.0), d * 1.5, vCenter.xyz);
        return min(min(a3.x, a3.y), a3.z);
      }


      void main()	{
       //this is for plane geometry
       vec2 olduv = gl_FragCoord.xy/resolution.xy ;

       vec2 uv = vUv ;
       vec2 newUv = uv + vec2( -utime*0.004, -utime*0.004);
       vec2 newUv2 = uv + vec2( utime*0.004, utime*0.004);


       float scale = 5.;
       vec2 originaluv = uv*5.;
       vec2 p = newUv*scale;
       vec2 p2 = newUv2*scale;

       float noiseoriginal = fbm( originaluv + fbm( originaluv ) )*.8;
       float noise = fbm( p + fbm( p ))*.8;
       float noise2 = fbm( p2 + fbm( p2 ))*.8;

       vec4 animatedNoise = vec4(noise) + vec4(noise2);

       vec4 totaloffset = animatedNoise + vec4(noiseoriginal);
       vec3 refracted = refract(eyeVector, vNormal, 1.0/2.42);
       vec2 p3 = vec2(uv.x +totaloffset.x, uv.y + totaloffset.y)*scale;
       float noise3 = fbm( p3 + fbm( p3 ))+.8;

       vec4 finalColor = vec4(noise3) * animatedNoise ;
       float tone = getTone(finalColor.rgb);
       vec3 rumpedCol = texture2D(colorRamp,vec2(tone, 0.)).rgb;      

       gl_FragColor =  vec4(rumpedCol,1.)*0.5;

      float f = Fresnel(eyeVector, vNormal);

       vec3 newCam = vec3(camPos.x,camPos.y,7.);
       vec3 viewDirectionW = normalize(camPos - vPosition);
       float fresnelTerm = dot(viewDirectionW, vNormal);  
       fresnelTerm = clamp( .5 - fresnelTerm, 0., 1.) ;
       gl_FragColor += fresnelTerm*0.6;
      //  gl_FragColor = dot(vNormal,vec3(0,0,0));
      vec2 uv2 = uv;
      vec3 pixeltex = texture2D(noiseTex,mod(uv*5.,1.)).rgb;      
      float iTime = utime*0.004;
      // camPos.x*0.04;
      uv.y += iTime / 10.0;
      uv.x -= (sin(iTime/10.0)/2.0);
      
      
      uv2.x += iTime / 14.0;
      uv2.x += (sin(iTime/10.0)/9.0);
      float result = 0.0;
      result += texture2D(noiseTex, mod(uv*4.,1.) * 0.6 + vec2(iTime*-0.003)).r;
      result *= texture2D(noiseTex, mod(uv2*4.,1.) * 0.9 + vec2(iTime*+0.002)).b;
      result = pow(result, 15.0);
      gl_FragColor += vec4(108.0)*result;

      gl_FragColor = mix(vec4(1., 1., 1., 0.15), gl_FragColor, edgeFactorTri());

      }

`;

var scene,
  camera,
  renderer,
  width = window.innerWidth,
  height = window.innerHeight,
  material,
  bloomPass,
  sphereobject,
  controls,
  key,
  h,
  composer;

const text1 = new THREE.TextureLoader().load('https://raw.githubusercontent.com/pizza3/asset/master/rgbnoise2.png');

var uniforms = {
  utime: {
    type: "f",
    value: 10.0,
  },
  colorRamp: {
    type: "t",
    value: new THREE.TextureLoader().load('https://raw.githubusercontent.com/pizza3/asset/master/color11.png'),
  },
  noiseTex: {
    type: "t",
    value: text1,
  },
  resolution: {
    value: new THREE.Vector2(width, height),
  },
  color1: {
    value: new THREE.Vector3(...options.color1),
  },
  color0: {
    value: new THREE.Vector3(...options.color0),
  },
};

material = new THREE.ShaderMaterial({
  uniforms: uniforms,
  transparent: true,
  vertexShader: vert,
  fragmentShader: frag,
});
var matrix = new THREE.Matrix4();
var period = 5;
var clock = new THREE.Clock();

function init() {
  createScene();
  plane();
  animate();
}
function createScene() {
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.z = 5;
  camera.position.y = 2.4;
  renderer = new THREE.WebGLRenderer();
  renderer.antialias = true;
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(width, height);
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  // renderer.interpolateneMapping = THREE.ACESFilmicToneMapping;
  // renderer.outputEncoding = THREE.sRGBEncoding;

  controls = new OrbitControls(camera, renderer.domElement);
  controls.update();
  controls.autoRotate = true;
  document.getElementById("world").appendChild(renderer.domElement);

  var renderScene = new RenderPass(scene, camera);

  bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    1.5,
    0.4,
    0.85
  );
  bloomPass.threshold = options.bloomThreshold;
  bloomPass.strength = options.bloomStrength;
  bloomPass.radius = options.bloomRadius;

  composer = new EffectComposer(renderer);
  composer.addPass(renderScene);
  composer.addPass(bloomPass);
}

function plane() {
  // var spheregeometry = new THREE.DodecahedronGeometry(0.5, 0);
  var spheregeometry = new THREE.SphereGeometry(1.7, 6, 3);

  var geometry = new THREE.BufferGeometry().fromGeometry(spheregeometry);
  geometry.setAttribute("center", {
    type: "v3",
    value: null,
    boundTo: "faceVertices",
  });
  geometry.computeVertexNormals();
  setUpBarycentricCoordinates(geometry);

  sphereobject = new THREE.Mesh(geometry, material);
  sphereobject.scale.set(1, 1.8, 1);

  scene.add(sphereobject);
}

function handleResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

let rot = 0.005;
function animate(delta) {
  requestAnimationFrame(animate);
  delta *= 0.01;
  material.uniforms.utime.value = delta;
  material.uniforms.color1.value = new THREE.Vector3(...options.color1);
  material.uniforms.color0.value = new THREE.Vector3(...options.color0);
  matrix.makeRotationY((clock.getDelta() * 0.5 * Math.PI) / period);
  camera.position.applyMatrix4(matrix);
  camera.lookAt(sphereobject.position);
  bloomPass.threshold = options.bloomThreshold;
  bloomPass.strength = options.bloomStrength;
  bloomPass.radius = options.bloomRadius;
  composer.render();
}
init();
window.addEventListener("resize", handleResize, false);

function setUpBarycentricCoordinates(geometry) {
  var positions = geometry.attributes.position.array;
  var normals = geometry.attributes.normal.array;

  // Build new attribute storing barycentric coordinates
  // for each vertex
  var centers = new THREE.BufferAttribute(
    new Float32Array(positions.length),
    3
  );
  // start with all edges disabled
  for (var f = 0; f < positions.length; f++) {
    centers.array[f] = 1;
  }
  geometry.setAttribute
	("center", centers);

  // Hash all the edges and remember which face they're associated with
  // (Adapted from THREE.EdgesHelper)
  function sortFunction(a, b) {
    if (a[0] - b[0] != 0) {
      return a[0] - b[0];
    } else if (a[1] - b[1] != 0) {
      return a[1] - b[1];
    } else {
      return a[2] - b[2];
    }
  }
  var edge = [0, 0];
  var hash = {};
  var face;
  var numEdges = 0;

  for (var i = 0; i < positions.length / 9; i++) {
    var a = i * 9;
    face = [
      [positions[a + 0], positions[a + 1], positions[a + 2]],
      [positions[a + 3], positions[a + 4], positions[a + 5]],
      [positions[a + 6], positions[a + 7], positions[a + 8]],
    ];
    for (var j = 0; j < 3; j++) {
      var k = (j + 1) % 3;
      var b = j * 3;
      var c = k * 3;
      edge[0] = face[j];
      edge[1] = face[k];
      edge.sort(sortFunction);
      key = edge[0] + " | " + edge[1];
      if (hash[key] == undefined) {
        hash[key] = {
          face1: a,
          face1vert1: a + b,
          face1vert2: a + c,
          face2: undefined,
          face2vert1: undefined,
          face2vert2: undefined,
        };
        numEdges++;
      } else {
        hash[key].face2 = a;
        hash[key].face2vert1 = a + b;
        hash[key].face2vert2 = a + c;
      }
    }
  }

  var index = 0;
  for (key in hash) {
    h = hash[key];

    // ditch any edges that are bordered by two coplanar faces
    var normal1, normal2;
    if (h.face2 !== undefined) {
      normal1 = new THREE.Vector3(
        normals[h.face1 + 0],
        normals[h.face1 + 1],
        normals[h.face1 + 2]
      );
      normal2 = new THREE.Vector3(
        normals[h.face2 + 0],
        normals[h.face2 + 1],
        normals[h.face2 + 2]
      );
      if (normal1.dot(normal2) >= 0.9999) {
        continue;
      }
    }

    // mark edge vertices as such by altering barycentric coordinates
    var otherVert;
    otherVert = 3 - ((h.face1vert1 / 3) % 3) - ((h.face1vert2 / 3) % 3);
    centers.array[h.face1vert1 + otherVert] = 0;
    centers.array[h.face1vert2 + otherVert] = 0;

    otherVert = 3 - ((h.face2vert1 / 3) % 3) - ((h.face2vert2 / 3) % 3);
    centers.array[h.face2vert1 + otherVert] = 0;
    centers.array[h.face2vert2 + otherVert] = 0;
  }
}
