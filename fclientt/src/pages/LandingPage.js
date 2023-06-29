import React, { useState, useEffect } from "react";
import { useLocation } from 'react-router-dom';
import httpClient from "../httpClient";
import Plot from "react-plotly.js";

const LandingPage = () => {
  const [datat, setData] = useState(null);
  const [plotdata, setPlotData] = useState(null);
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const [gasprice, setPrice] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      try {
        const resp = await httpClient.get("//localhost:5000/@data");
        setData(resp.data);
      } catch (error) {
        console.error('Error parsing response data:', error);
      }
    };

    fetchData();
  }, []);

  useEffect(() => {
    if (datat) {
      const createPlotData = () => {
        const plotData = [
          {
            x: datat.current_datetime,
            y: datat.gas_price_Gwei,
            type: 'scatter',
            mode: 'lines',
            marker: { color: 'blue' },
          },
        ];
        setPlotData(plotData);
      };

      createPlotData();
    }
  }, [datat]);


  useEffect(() => {
    (async () => {
      try {
        const resp = await httpClient.get("//localhost:5000/gasprice");
        setPrice(resp.data);
      } catch (error) {
        console.log("Not available");
      }
    })();
    const intervalId = setInterval(async () => {
      try {
        const resp = await httpClient.get("//localhost:5000/gasprice");
        setPrice(resp.data);
      } catch (error) {
        console.log("Not available");
      }
    }, 2000); 
    return () => clearInterval(intervalId);
  }, []);







  return (
    <div>
      <div class="row">
{/* <div class="col md-2">
      <div class="sidebar-sticky">
            <ul class="nav flex-column">
              <li class="nav-item">
                <a class="nav-link active" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-home"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
                  Dashboard <span class="sr-only">(current)</span>
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-file"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path><polyline points="13 2 13 9 20 9"></polyline></svg>
                  Orders
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-shopping-cart"><circle cx="9" cy="21" r="1"></circle><circle cx="20" cy="21" r="1"></circle><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"></path></svg>
                  Products
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-users"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
                  Customers
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bar-chart-2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                  Reports
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-layers"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
                  Integrations
                </a>
              </li>
            </ul>

            <h6 class="sidebar-heading d-flex justify-content-between align-items-center px-3 mt-4 mb-1 text-muted">
              <span>Saved reports</span>
              <a class="d-flex align-items-center text-muted" href="#">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-plus-circle"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
              </a>
            </h6>
            <ul class="nav flex-column mb-2">
              <li class="nav-item">
                <a class="nav-link" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-file-text"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                  Current month
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-file-text"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                  Last quarter
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-file-text"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                  Social engagement
                </a>
              </li>
              <li class="nav-item">
                <a class="nav-link" href="#">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-file-text"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                  Year-end sale
                </a>
              </li>
            </ul>
          </div>
</div> */}
 {/* <div class="col md-3">

  </div>   */}

  {/* <div class="col">
  </div>  */}
  <div class="col md-16 "> 
  <div class="card">

<div class="card-body">

<div class="row justify-content-between mb-3">

<div class="col col-md-auto text-center text-md-end">

<div class="d-none" id="divTimestamp">
<a id="btnViewSnapshot" class="btn btn-soft-secondary d-none" onclick="switchToLive();" href="javascript:;"><i class="fas fa-arrow-left me-1"></i><span id="spanSnapshot">View Live</span></a>
<span id="spanTimestamp" class="d-block d-sm-inline-block text-muted mt-2 mt-sm-0"></span>
</div>
</div>
</div>


<div class="row text-center mb-4">
<div class="col-md-4 mb-1 mb-md-0">
<div class="card h-100 shadow-none p-3" id="divLow">
<i class="bi bi-arrow-down-right-circle-fill">
<svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" class="bi bi-arrow-down-right-circle-fill" viewBox="0 0 16 16">
  <path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm5.904-2.803a.5.5 0 1 0-.707.707L9.293 10H6.525a.5.5 0 0 0 0 1H10.5a.5.5 0 0 0 .5-.5V6.525a.5.5 0 0 0-1 0v2.768L5.904 5.197z"/>
</svg>
</i>
<div class="h4 mb-2 fw-normal"><font color="green"><span id="spanHighPrice"> <h4 class="h6 mb-2">Low</h4></span></font></div>
<div id="divLowPrice" class="">
<div class="h4 text-success fw-normal mb-2">
<span id="spanLowPrice">{parseInt(gasprice.safe_gas_price)}</span> gwei
</div>
<div class="text-muted">
<span id="spanLowPriorityAndBase">Base:{parseInt(gasprice.BaseFee)}  | Priority: {parseInt(gasprice.priority_safe)} </span>
</div>

</div>
<div id="divLowSnapshot" class="d-none">
<div class="h4 text-success mb-1">
<span id="spanLowPriceSnapshot">
<span id="ContentPlaceHolder1_ltGasPriceSnapshot"></span></span> gwei
</div>
<div class="text-muted">
<span id="spanLowPriorityAndBaseSnapshot">
</span>
</div>
<div class="text-muted">
</div>
</div>
</div>
</div>
<div class="col-md-4 mb-1 mb-md-0">
<div class="card h-100 shadow-none p-3" id="divAvg">
<i class="bi bi-arrow-right-circle-fill">
<svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" class="bi bi-arrow-right-circle-fill" viewBox="0 0 16 16">
  <path d="M8 0a8 8 0 1 1 0 16A8 8 0 0 1 8 0zM4.5 7.5a.5.5 0 0 0 0 1h5.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H4.5z"/>
</svg>
</i>
<div class="h4 mb-2 fw-normal"><font color="blue"><span id="spanHighPrice"> <h4 class="h6 mb-2">Average</h4></span></font></div>

<div id="divAvgPrice" class="">
<div class="h4 text-primary fw-normal mb-2"><span id="spanAvgPrice">{parseInt(gasprice.ProposeGasPrice)}</span> gwei</div>
<div class="text-muted">
<span id="spanProposePriorityAndBase">Base:{parseInt(gasprice.BaseFee)} | Priority: {parseInt(gasprice.priority_propose)}</span>
</div>

</div>
<div id="divAvgPriceSnapshot" class="d-none">
<span class="h4 text-primary mb-1">
<span id="spanAvgPriceSnapshot">
</span> gwei</span>
<div class="text-muted">
<span id="spanProposePriorityAndBaseSnapshot">
</span>
</div>
<div class="text-muted">
</div>
</div>
</div>
</div>
<div class="col-md-4">
<div class="card h-100 shadow-none p-3" id="divHigh">
<i class="bi bi-arrow-up-right-circle-fill">
<svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" class="bi bi-arrow-up-right-circle-fill" viewBox="0 0 16 16">
  <path d="M0 8a8 8 0 1 0 16 0A8 8 0 0 0 0 8zm5.904 2.803a.5.5 0 1 1-.707-.707L9.293 6H6.525a.5.5 0 1 1 0-1H10.5a.5.5 0 0 1 .5.5v3.975a.5.5 0 0 1-1 0V6.707l-4.096 4.096z"/>
</svg>
</i>
<div class="h4 mb-2 fw-normal"><font color="red"><span id="spanHighPrice"> <h4 class="h6 mb-2">High</h4></span></font></div>
<div id="divHighPrice" class="">
<div class="h4 mb-2 fw-normal"><font color="brown"><span id="spanHighPrice">{parseInt(gasprice.fast_gas_price)}</span> gwei</font></div>
<div class="text-muted">
<span id="spanHighPriorityAndBase">Base: {parseInt(gasprice.BaseFee)} | Priority:{parseInt(gasprice.priority_fast)} </span>
</div>
</div>
{/* <div id="divHighPriceSnapshot" class="d-none">
<div class="h4 mb-0"><font color="brown"><span id="spanHighPriceSnapshot">
</span> gwei</font></div>
<div class="text-muted">
<span id="spanHighPriorityAndBaseSnapshot">
</span>
</div>
<div class="text-muted">
</div>
</div> */}
</div>
</div>
</div>



</div>

&nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   

  {plotdata && (
      <Plot
        data={plotdata}
        layout={{ title: 'Gas price distribution' }}
        config={{ responsive: true }}
      />
     )}
 
  {/* <div class="col md-8">
  {plotdata && (
      <Plot
        data={plotdata}
        layout={{ title: 'Gas price distribution' }}
        config={{ responsive: true }}
      />
     )}
  </div> */}
</div>



  </div>   
 

  {/* <div class="col">
  </div>   */}

   
  
  </div>
  </div>
  );  
};

export default LandingPage;