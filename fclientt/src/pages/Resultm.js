import React, { useState, useEffect } from "react";
import { useLocation } from 'react-router-dom';
import Plot from "react-plotly.js";

export const Resultm = () => {

  const [pred_M, setPredm] = useState([]);
  const [dates_M, setDatesm] = useState([]);
  const [nbr, setNbr] = useState([]);
  const [currenttime, setCurrenttime] = useState([]);
  const [minindexm, setMinindexm] = useState(null);
  const [mindatem, setMindatem] = useState(null);
  const [maxindexm, setMaxindexm] = useState(null);
  const [maxdatem, setMaxdatem] = useState(null);
  const [average, setAverage] = useState(null);
  const [plotdatam, setPlotDatam] = useState(null);
  const [besttime, setBesttime] = useState(null);
  const location = useLocation();

  const queryParams = new URLSearchParams(location.search);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const responseData = JSON.parse(queryParams.get('response').substring(1));

        const pred_M = responseData.predictions_M;
        const dates_M = responseData.dates_M;
        const nbr=responseData.nbr;

        setPredm(pred_M);
        setDatesm(dates_M);
        setNbr(nbr);


        const minIndexm = pred_M.indexOf(Math.min(...pred_M));
        const minDatem = dates_M[minIndexm];

        const currentTimestamp = Date.now();

        const maxIndexm = pred_M.indexOf(Math.max(...pred_M));
        const maxDatem = dates_M[maxIndexm];
        
        const averagePred =pred_M.reduce((total, num) => total + num, 0) / pred_M.length;


        setMinindexm(minIndexm);
        setMindatem(minDatem);
        setMaxindexm(maxIndexm);
        setMaxdatem(maxDatem);
        setAverage(averagePred);
        setCurrenttime(currentTimestamp);

        const datam = [
          {
             x: dates_M,
            y: pred_M,
            type: 'scatter',
            mode: 'lines',
            marker: { color: 'red' },
          },
        ];

        setPlotDatam(datam);
     
      } catch (error) {
        console.error('Error parsing response data:', error);
      }
    };

    fetchData();
  }, []);
  useEffect(() => {
    if (currenttime,mindatem) {
      const calculateBesttime = () => {
        const minDate = new Date(mindatem); 
        const differenceInMilliseconds =minDate.getTime()- currenttime  ;
        const differenceInSeconds =( Math.floor(differenceInMilliseconds / 1000))-3600;
        // const hours = Math.floor(differenceInSeconds / 3600);
        const minutes = Math.floor((differenceInSeconds % 3600) / 60);
        const seconds = Math.floor(differenceInSeconds % 60);
        const bestTime = `${minutes}M:${seconds}S`;
        setBesttime(bestTime)
      };

      calculateBesttime();
    }
  }, [currenttime,mindatem]);

  return (


    <div>
    <br/>
   


<div class="container">

<br/>
<button type="button" class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#staticBackdrop">
  View forecasted Results
</button>


<div class="modal fade" id="staticBackdrop" data-bs-backdrop="static" data-bs-keyboard="false" tabindex="-1" aria-labelledby="staticBackdropLabel" aria-hidden="true">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <h1 class="modal-title h5" id="staticBackdropLabel"><div class="h4 mb-2 fw-normal"><font color="Black"><span id="spanHighPrice"><h3 class="panel-title">Prediction Results</h3> </span> </font></div> </h1>
        
        <button type="button" class="btn-close" data-bs-dismiss="modal"><span class="visually-hidden">Close</span></button>
      </div>
      <div class="modal-body">
      <div class="panel panel-data">
<div class="panel-header">

    </div>
    <div class="panel-content">

    <table class="table">
        <thead>

            <tr>
            <th><div class="h4 mb-2 fw-normal"><font color="Black"><span id="spanHighPrice"><h3 class="panel-title">Date</h3> </span> </font></div> 
               </th>
               <th><div class="h4 mb-2 fw-normal"><font color="Black"><span id="spanHighPrice"><h3 class="panel-title">Value</h3> </span> </font></div> 
               </th>
            </tr>
            </thead>


        {dates_M.map((date, index) => (
           <div key={index}>
            


              <tbody>
                
                <tr>
                    <td>{date}</td>
                    &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; 
                    <td >
                    {pred_M[index].toFixed(3)}
                        
                    </td>
                </tr>

        </tbody>

       
         </div> 
        ))}

        </table>
    </div> 

    
</div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-primary" data-bs-dismiss="modal">Close</button>
      </div>
    </div>
  </div>
</div>




<br/>

<br/>
<h3 class="panel-title">Prediction in {nbr} Minutes</h3>
                    <div class="page-tabs tab-content">
                        <div class="tab-pane active">
                            
                                
    <div class="page-body">

<div class="row">
<div class="col-md-8">
<div>
      
        <div>  
    {plotdatam && (
      <Plot
        data={plotdatam}
        layout={{ title: 'Predictions in Minnutes' }}
        config={{ responsive: true }}
      />
     )}
     </div>
     </div>

     <br/><br/><br/> <br/><br/><br/>


</div>
<div class="col-md-4">
<div class="row">
<div class="col-6 col-md-12">
<table class="table">
<tbody>

<tr>
                            <td><div class="h4 mb-2 fw-normal"><font color="black"><span id="spanHighPrice"> Best Time :</span> </font></div> </td>
                            <td><div class="h4 mb-2 fw-normal"><font color="blue"><span id="spanHighPrice">After {besttime}</span> </font></div> </td>
                        </tr>
                    
                        
                        <tr>
                          
                            <td><div class="h4 mb-2 fw-normal"><font color="black"><span id="spanHighPrice"> Minimum Date: </span> </font></div> </td>
                            <td><div class="h4 mb-2 fw-normal"><font color="green"><span id="spanHighPrice">{mindatem}</span> </font></div> </td>
                    
                        </tr>
                        <tr >
                          
                            <td><div class="h4 mb-2 fw-normal"><font color="black"><span id="spanHighPrice"> Minimum Gas Price: </span> </font></div> </td>
                            <td><div class="h4 mb-2 fw-normal"><font color="green"><span id="spanHighPrice">{pred_M[minindexm]}</span> </font></div> </td>
                         
                        </tr>
                    
                    <tr>
                       
                        <td><div class="h4 mb-2 fw-normal"><font color="black"><span id="spanHighPrice"> Maximum Date:</span> </font></div> </td>
                        <td><div class="h4 mb-2 fw-normal"><font color="red"><span id="spanHighPrice">{maxdatem}</span> </font></div> </td>
                      
                    </tr>
                    
                    <tr>
                    <td><div class="h4 mb-2 fw-normal"><font color="black"><span id="spanHighPrice"> Maximum Gas Price: </span> </font></div> </td>
                       
                        <td><div class="h4 mb-2 fw-normal"><font color="red"><span id="spanHighPrice">{pred_M[maxindexm]}</span> </font></div> </td>
                        
                    </tr>

                        <tr>
                            <td><div class="h4 mb-2 fw-normal"><font color="black"><span id="spanHighPrice"> Average :</span> </font></div> </td>
                            <td><div class="h4 mb-2 fw-normal"><font color="orange"><span id="spanHighPrice"> {average}</span> </font></div> </td>
                        
                        </tr>
                       
                </tbody>
                </table>

</div>

</div>


</div>

</div>
</div>
</div>
</div>
</div>
    
    
   
</div>
  
     );
};

export default Resultm;



