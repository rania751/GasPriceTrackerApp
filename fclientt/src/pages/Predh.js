import React, { useState } from "react";
import httpClient from "../httpClient";


export const Predh = () => {


    const [nbr, setNbr] = useState("");
    // const [nbrm, setNbrm] = useState("");
    const [snipper, setSnipper] = useState(true);
    var nbrint
    var nbrintm
    const sendNbr = async () => {
          try {
            nbrint=parseInt(nbr)
            setSnipper(false)

            const resp = await httpClient.post("//localhost:5000/predict", {
              nbrint
            });
            
            window.location.href = `/Account/Result?response==${encodeURIComponent(JSON.stringify(resp.data))}`;
            
          
          } catch (error) {
            if (error.response.status === 401) {
              alert("Invalid input");
            } if (error.response.status === 431) {
              alert("Request Header Fields Too Large");
              window.location.href = "/Account";

            }
          }
        };
        // const sendNbrm = async () => {
        //   try {
        //     nbrintm=parseInt(nbrm)
        //     setSnipper(false)
        //     const resp = await httpClient.post("//localhost:5000/predictm", {
        //       nbrintm
        //     });
            
        //     console.log(snipper)
        //     window.location.href = `/Account/Resultm?response==${encodeURIComponent(JSON.stringify(resp.data))}`;
            
        //     console.log(resp)
        //   } catch (error) {
        //     if (error.response.status === 401) {
        //        alert("Invalid input");
        //     }
        //     if (error.response.status === 431) {
        //       alert("Request Header Fields Too Large");
        //       window.location.href = "/Account";
  
        //     }

        //   }
        // };

  return (
    <div>
      
    <div className="container">
    <div class="row">

    <div class="col md-6"></div>
    <div class="card h-100 shadow-none p-3" id="divLow">
<div class="container text-center">

<div class="row align-items-center">
 <div class="col">
 {snipper ? ( <div class="d-flex flex-column align-items-center">
     <h3>Prediction in hours</h3><p>(Max number=180)</p>
     <input class="form-control" type="text" value={nbr} style={{ width: '200px', fontSize: '14px' }} onChange={(e) => setNbr(e.target.value)}  aria-label="default input example"></input>
     <button type="button" class="btn btn-primary mt-2" onClick={() => sendNbr()}>
Submit
</button></div>):(<div>
<br/><br/><br/><br/><br/>
<div class="row">
 {/* <div class="col"></div>
 */}
 <div class="col"></div> 
 <div class="col">
         <div class="spinner-grow text-primary" role="status">
<span class="visually-hidden">Loading...</span>
</div>
<div class="spinner-grow text-secondary" role="status">
<span class="visually-hidden">Loading...</span>
</div>
</div>
<div class="col"></div>
       </div>
     <br/><br/>

</div>)}

 </div>

 {/* <div class="col">
 {snipper ? ( <div>
    <div> <h3>Prediction in minutes  </h3> <span><p>(Max number=180)</p></span></div>
    
     <input class="form-control" type="text" value={nbrm} onChange={(e) => setNbrm(e.target.value)} aria-label="default input example"></input>
     <button type="button" class="btn btn-primary mt-2" onClick={() => sendNbrm()}>
Submit
</button></div>):(<div>
<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
<div class="row">
 <div class="col"></div>
 <div class="col"></div>
 <div class="col">
         <div class="spinner-grow text-primary" role="status">
<span class="visually-hidden">Loading...</span>
</div>
<div class="spinner-grow text-secondary" role="status">
<span class="visually-hidden">Loading...</span>
</div>
</div>
<div class="col"></div>
       </div>
       <br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

</div>)}
 </div> */}
</div>
<br/><br/>
</div>

</div>
 </div>
 </div>
 </div>
  )
}
export default Predh;

