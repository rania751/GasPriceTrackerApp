import axios from "axios";
import React, { useState, useEffect } from "react";

import Plot from "react-plotly.js";

export const Admin = () => {
  const [metric, setMetrics] = useState(null);
  const [test, setTestdata] = useState(null);
  const [testall, setTestdataall] = useState(null);
  const [plotforcast, setPlotfor] = useState(null);
  const [trainplot, setPlottrain] = useState(null);
  const [pred, setPred] = useState([]);
  const [actual, setActual] = useState([]);
  const [plotData, setPlotData] = useState(null);
  const [dates, setDates] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const resp = await axios.get("//localhost:5000/eval");
        const test = resp.data.plottest;
        setTestdata(test);
        const testall = resp.data.plotall;
        setTestdataall(testall);
        const plotforcast = resp.data.plotfor;
        setPlotfor(plotforcast);
        const trainplot = resp.data.trainplott;
        setPlottrain(trainplot);
        const pred1 = resp.data.testp;
        const actual1 = resp.data.testa;
        const dates1 = resp.data.datestest;
        setPred(pred1);
        setActual(actual1);
        setDates(dates1);
        setMetrics(resp.data);
        setPlotData(JSON.parse(resp.data.plot_data1));
      } catch (error) {
        console.log("Not authenticated");
      }
    };

    fetchData();
  }, []);

  return (
    <div>
      {metric ? (
        <>
          <br/><br/><br/>
          <div className="container">
            <div className="row">
              <div className="col md-6"></div>
              <button type="button" class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#staticBackdrop">
  View forecasted Results
</button>
<div class="modal fade" id="staticBackdrop" data-bs-backdrop="static" data-bs-keyboard="false" tabindex="-1" aria-labelledby="staticBackdropLabel" aria-hidden="true">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <h1 class="modal-title h5" id="staticBackdropLabel"><div class="h4 mb-3 fw-normal"><font color="Black"><span id="spanHighPrice"><h3 class="panel-title">Prediction Results</h3> </span> </font></div> </h1>
        
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
            <th><div class="h4 mb-3 fw-normal"><font color="Black"><span id="spanHighPrice"><h3 class="panel-title">Predected</h3> </span> </font></div> 
               </th>
               <th><div class="h4 mb-3 fw-normal"><font color="Black"><span id="spanHighPrice"><h3 class="panel-title">Actual</h3> </span> </font></div> 
               </th>
               {/* <th><div class="h4 mb-3 fw-normal"><font color="Black"><span id="spanHighPrice"><h3 class="panel-title">actual</h3> </span> </font></div> 
               </th> */}

            </tr>
            </thead>


        {pred.map((pred, index) => (
           <div key={index}>
            

              <tbody>
             
                
                <tr>
                    <td>{pred}</td> 
                     &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; 
                    <td >
                    {actual[index]}
                        
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

              <div className="card h-100 shadow-none p-3" id="divLow">
                <div className="container text-center">
                  <div className="row align-items-center">


                      <div className="col">
                      <table className="table table-dark table-sm">
                        <caption className="visually-hidden">
                          Boosted tables basic look
                        </caption>
                        <tbody>
                          <tr>
                            <th scope="row">MAE</th>
                            <td>{metric.mae}</td>
                          </tr>
                          <tr>
                            <th scope="row">MSE</th>
                            <td>{metric.mse}</td>
                          </tr>
                          <tr>
                            <th scope="row">RMSE</th>
                            <td>{metric.rmse}</td>
                          </tr>
                          <tr>
                            <th scope="row">MAPE_Train</th>
                            <td>{metric.mape_train}</td>
                          </tr>
                          <tr>
                            <th scope="row">MAPE_Test</th>
                            <td>{metric.mape_test}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>



                    <div class="col">
<div>
      {plotData ? (
        <Plot
          data={plotData.data}
          layout={plotData.layout}
          config={{ displayModeBar: false }}
        />
      ) : (
        <p>Loading plot data...</p>
      )}
    </div>

</div>


<div class="container">
  <div class="row">
   


<div className="col">
                      <div>
                      {test ? (
                              <Plot
                                data={test}
                                layout={{ title: "Test and Forecaste " }}
                                config={{ displayModeBar: false }}
                              />
                            ) : (
                          <p>Loading plot data...</p>
                        )}
                      </div>
                    </div>

<div className="col">
                      <div>
              
                      {trainplot ? (
                              <Plot
                                data={trainplot}
                                layout={{ title: "Train and Forecast" }}
                                config={{ displayModeBar: false }}
                              />
                            ) : (
                          <p>Loading plot data...</p>
                        )}
                      </div>
                    </div>
                    <div className="col">
                      <div>
                      {plotforcast ? (
                              <Plot
                                data={plotforcast}
                                layout={{ title: "Forecaste Plot" }}
                                config={{ displayModeBar: false }}
                              />
                            ) : (
                          <p>Loading plot data...</p>
                        )}
                      </div>
                    </div>
                    <div className="col">
                      <div>
                      {testall ? (
                              <Plot
                                data={testall}
                                layout={{ title: "Dataset and forecaste " }}
                                config={{ displayModeBar: false }}
                              />
                            ) : (
                          <p>Loading plot data...</p>
                        )}
                      </div>
                    </div>
                    </div>
</div>
                  </div>
                </div>
              </div>
              
              <div className="col"></div>
            </div>
          </div>
        </>
      ) : (
        <div>
          <br/><br/><br/>
          <div className="container">
            <br/><br/><br/><br/>
            <div className="row">
              <div className="col md-6"></div>
              <div className="card h-100 shadow-none p-3" id="divLow">
                <div className="container text-center">
                  <br/><br/><br/><br/>
                  <div className="row align-items-center">
                    <div className="col"></div>
                    <div className="col">
                      <div className="spinner-grow text-primary" role="status">
                        <span className="visually-hidden">Loading...</span>
                      </div>
                      <div className="spinner-grow text-secondary" role="status">
                        <span className="visually-hidden">Loading...</span>
                      </div>
                    </div>
                    <div className="col"></div>
                  </div>
                  <br/><br/><br/><br/><br/>
                </div>
              </div>
            </div>
          </div>
        </div>
        

      )}
    
    </div>
    
  );
};
export default Admin;

