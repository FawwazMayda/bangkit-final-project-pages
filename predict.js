let model;
const classes = ["healthy","multiple diseases","rust","scab"]
$(document).ready()
{

  $('.progress-bar').hide();
}

$("#image-selector").change(function(){
    let reader = new FileReader();
    console.log("New Image")
    reader.onload = function(){
        let dataURL = reader.result;
        $("#selected-image").attr("src",dataURL);
        $("#prediction-list").empty();
    }
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});


$("#model-selector").change(function(){
    loadModel($("#model-selector").val());
    $('.progress-bar').show();
})

async function loadModel(name){
    //model=await tf.loadLayersModel(`http://localhost:443/${name}/model.json`);
    
    model=await tf.loadLayersModel(`https://bangkit-final-server-node.et.r.appspot.com/${name}/model.json`) 
    console.log(`Loading ${name} model`)
    $('.progress-bar').hide();
}


$("#predict-button").click(async function(){
  // console.log("inside the vgg preProcessing:");
    let image= $('#selected-image').get(0);

    let tensor = preprocessImage(image,$("#model-selector").val());

    let prediction = await model.predict(tensor).data();
    let top5=Array.from(prediction)
                .map(function(p,i){
    return {
        probability: p,
        className: classes[i]
    };
    }).sort(function(a,b){
        return b.probability-a.probability;
    }).slice(0,5);

$("#prediction-list").empty();
top5.forEach(function(p){
    $("#prediction-list").append(`<li>${p.className}:${p.probability.toFixed(6)}</li>`);
});

});


function preprocessImage(image,modelName)
{
    let tensor=tf.browser.fromPixels(image)
    .resizeNearestNeighbor([224,224])
    .toFloat();//.sub(meanImageNetRGB)

    if(modelName==undefined)
    {
        return tensor.expandDims();
    }
    else if(modelName=="vgg16" || modelName=="resnet")
    {
        console.log("inside the vgg preProcessing:");
        let meanImageNetRGB= tf.tensor1d([123.68,116.779,103.939]);
        return tensor.sub(meanImageNetRGB)
                    .reverse(2)   // using the conventions from vgg16 documentation
                    .expandDims();
          
    }
    else if(modelName=="efficientnet")
    {
        let offset=tf.scalar(127.5);
        return tensor.sub(offset)
                    .div(offset)
                    .expandDims();
    }
    else
    {
        throw new Error("UnKnown Model error");
    }
}