import p5 from 'p5';
import ml5 from 'ml5';

let video;
let features;
let knn;
let labelP;
let ready = false;
let x;
let y;
let label = 'nothing';

const App = (App) => 
{

    App.setup = () => 
    {
        App.createCanvas();
        video = App.createCapture(App.VIDEO);
        video.size(320,240);
        features = ml5.featureExtractor("MobileNet", modelReady);
        knn = ml5.KNNClassifier();
        labelP = App.createP('need training data');
        labelP.style("font-size", "32pt");
        x = width/2;
        y = height/2;
    }

    App.draw()
    {
        App.background(0);
        App.fill(255);
        App.ellipse(x,y,24);

        if (label == 'left') 
        {
            x--;
        } 
        else if (label == 'right') 
        {
            x++;
        } 
        else if (label == 'up') 
        {
            y--;
        } 
        else if (label == 'down') 
        {
            y++;
        }
        
        //image(video, 0, 0);
        if (!ready && knn.getNumLabels() > 0) 
        {
            goClassify();
            ready = true;
        }
    }

    function goGlassify()
    {
        const logits = features.infer(video);
        knn.classify(logits, (error, result) => {
            if(error)
            {
                console.error(error);
            }
            else
            {
                label = result.label;
                labelP.html(result.label);
                goGlassify();
            }
        });
    }

    function modelReady()
    {
        console.log('modelReady');
    }

    function keyPressed()
    {
        const logits = features.infer(video);
        if (key == 'l') 
        {
            knn.addExample(logits, 'left');
            console.log('left');
        } 
        else if (key == 'r') 
        {
            knn.addExample(logits, 'right');
            console.log('right');
        } 
        else if (key == 'u') {
            knn.addExample(logits, 'up');
            console.log('up');
        } 
        else if (key == 'd') {
            knn.addExample(logits, 'down');
            console.log('down');
        } 
        else if (key == 's') 
        {
            //save(knn, 'model.json');
            knn.save('model.json');
        }
    }

}

new p5(App);