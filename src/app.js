import p5 from 'p5';
import ml5 from 'ml5';

//p5 declarations
let video;
let features;
let knn;
let labelP;
let x;
let y;
let label = 'nothing';
let myp5;
let padding = 15;
let numOfExamplesAdded = 100;
let btnHeight = 11;
const canvasDimensions = {width: 720, height: 405}

//Button Declaration
let leftButton, rightButton, upButton, downButton, saveButton;
const buttonArray = [];

const App = (App) => 
{

    /**
    * p5 Initialization Function.
    */
    
    App.setup = () => 
    {
        //Canvas and Framerate setup
        App.createCanvas(canvasDimensions.width, canvasDimensions.height);

        //Video creation
        video = App.createCapture(App.VIDEO);
        video.hide()
        video.size(canvasDimensions.width, canvasDimensions.height);

        //ML SETUP
        features = ml5.featureExtractor("MobileNet", modelReady);
        knn = ml5.KNNClassifier();

        //LABEL CREATION
        labelP = App.createP('need training data');
        labelP.style("font-size", "18pt");

        //CIRCLE POSITION
        x = App.width/2;
        y = App.height/2;

        //Button Creation
        leftButton = new Button('Train Left', 'Left_Button', {x:0 ,y:canvasDimensions.height+btnHeight},buttonArray).InstantiateButton();
        rightButton = new Button('Train Right', 'Right_Button', {x:leftButton.width+padding ,y:canvasDimensions.height+btnHeight},buttonArray).InstantiateButton();
        upButton = new Button('Train Up', 'Up_Button', {x:(rightButton.width+rightButton.x)+padding ,y:canvasDimensions.height+btnHeight},buttonArray).InstantiateButton();
        downButton = new Button('Train Down', 'Down_Button', {x:(upButton.width+upButton.x)+padding ,y:canvasDimensions.height+btnHeight},buttonArray).InstantiateButton();
        saveButton = new Button('Save Trained Model', 'Save_Button', {x:(downButton.width+downButton.x)+padding ,y:canvasDimensions.height+btnHeight}).InstantiateButton();
        SetButtonEvents(buttonArray,StartTraining);
        saveButton.mousePressed(SaveModel);
    }


    /**
    * p5 "ticker" function runs every frame.
    */
    
    App.draw = () =>
    {
        //FOR ELLISE DRAW
        App.background(0);
        App.fill(255);
        App.ellipse(x,y,24);

         // FLIPPING VIDEO CAPTURE
         App.push();
         App.translate(canvasDimensions.width,0);
         App.scale(-1,1);
         App.image(video, 0, 0, canvasDimensions.width, canvasDimensions.height);
         App.pop();


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

        /**
        * Would run the video if ready was false, and knn had labels trainied.
        */
        
        // if (!ready && knn.getNumLabels() > 0) 
        // {
        //     goClassify();
        //     ready = true;
        // }
    }


    /**
    * Goes through each button in the array and adds the specified event to.\
    * Will probably add this method to the initial class creation for buttons
    * @param {array} array The array of buttons you want the event added to.
    * @param {eventName} eventName Name of the function you want attached.
    */
    
    function SetButtonEvents(array, eventName)
    {
        array.forEach((btn) => {
            btn.mousePressed(eventName);
        });
    }


    /**
    * Saves model to a json file.
    *
    * @return {file} Downloads the trainied Model to the users specified browsers download location.
    */
    
    function SaveModel()
    {
        knn.save('model.json');
        labelP.html('Saving your trained model, please do not leave this webpage until you see that your model has downloaded from your browser.\nThank you.');
    }


    /**
    * async function that waits for a timer to complete
    */
    
    async function StartTraining()
    {
        let currentTarget = this.name;
        let label = currentTarget.split('_').shift().toLowerCase();
        const extractionReady = await TimerComplete();

        if(extractionReady)
        {
            labelP.html('START EXTRACTION');
            ExtractExamples(label, ExtractionFinished);        
        }
    }

    /**
    * Returns a resolved promise.
    *
    * @return {boolean} Returns true once the timer has completed.
    */
    
    function TimerComplete()
    {
        
        let trainingTime = 5;

        return new Promise(resolve =>
        {
            let interval = setInterval(() => {
                labelP.html(`Time until extraction: ${trainingTime.toString()}`);

                if(trainingTime == 0)
                {
                    resolve(true);
                    clearInterval(interval);
                }

                trainingTime--;
                    
            }, 1000);
        });
    }

    /**
    * Classifies whats in the video after its been trained.
    *
    * @return {results} Returns an object of results.
    */
    
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

    
    /**
    * Starts extracting examples
    *
    * @param {string} label Label that's being trained.
    * @param {number} examplesExtracted Current loop in the iteration.
    * @return {callback} Callback to let the user know its completed and they can proceed with the next label or save.
    */

    function ExtractExamples(label,callback)
    {
        const logits = features.infer(video);
        let examplesExtracted = 0;
        let delay;


        for(examplesExtracted; examplesExtracted != numOfExamplesAdded; examplesExtracted++)
            delay = setTimeout(() => { knn.addExample(logits, label); }, 500);

        if(examplesExtracted == numOfExamplesAdded)
        {
            callback(label);
            clearTimeout(delay);
        }
            
    }


    /**
    * Callback function to let the user no extraction is finished and they can
    * begin extracting examples for a new label
    *
    * @return {number} x raised to the n-th power.
    */
    
    function ExtractionFinished(label)
    {
        labelP.html(`Extraction for label: ${label} Has completed.\nMove onto your next label or save this model.`)
    }
    

    /**
    * Callback for when the model is loaded.
    *
    * @return {string} Updates html element to let the user know it's ready.
    */
    
    function modelReady()
    {
        labelP.html('Model is now ready, please press one of the buttons above to begin training.');
    }
}

myp5 = new p5(App);


/**
* Class to easily create and manage a p5 button
*
* @param {text} text String for button text.
* @param {name} name Name of the button for reference.
* @param {position} position Where to position the button on the webpage.
* @param {array} array Array to push the button to for easy management.
* @return {button} Created p5 button.
*/

class Button
{
    constructor(text, name, position, array=undefined)
    {
        this.text = text;
        this.name = name;
        this.position = {x: position.x, y: position.y};
        this.array = array;
    }

    InstantiateButton()
    {
        let btn = myp5.createButton(this.text);
        btn.name = this.name;
        btn.position(this.position.x, this.position.y);
        this.array != undefined ? this.array.push(btn) : console.warn('No array was proved to add list of buttons to.');
        return btn;
    }
}