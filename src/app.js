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

    function goClassify()
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
                goClassify();
            }
        });
    }

    function modelReady()
    {
        console.log('modelReady');
    }

    App.keyPressed = () =>
    {
        const logits = features.infer(video);
        if (App.key == 'a') 
        {
            knn.addExample(logits, 'left');
            console.log('left');
        } 
        else if (App.key == 'd') 
        {
            knn.addExample(logits, 'right');
            console.log('right');
        } 
        else if (App.key == 'w') {
            knn.addExample(logits, 'up');
            console.log('up');
        } 
        else if (App.key == 's') {
            knn.addExample(logits, 'down');
            console.log('down');
        } 
        else if (App.key == 'e') 
        {
            //save(knn, 'model.json');
            knn.save('model.json');
        }
    }

}

new p5(App);