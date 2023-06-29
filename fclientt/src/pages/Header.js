import React, { useState, useEffect } from "react";
import httpClient from "../httpClient";

export const Header = () => {

  const [currentTimestamp, setCurrentTimestamp] = useState(Date.now());
  const [gasprice, setPrice] = useState("");

  useEffect(() => {
    const intervalId = setInterval(() => {
      setCurrentTimestamp(Date.now());
    }, 1000);
    return () => clearInterval(intervalId);
  }, []);
  
const options = {
  weekday: "short",
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  timeZone: "UTC"
};

  const formattedTimestamp = new Date(currentTimestamp).toLocaleString("en-US", options);
    const [user, setUser] = useState(null);
    const [emailid,setEmailid]=useState("")



    const logoutUser = async () => {
    await httpClient.post("//localhost:5000/logout");
    window.location.href = "/";
      };
    
      useEffect(() => {
        (async () => {
          try {
            const resp = await httpClient.get("//localhost:5000/@me");
            setUser(resp.data);
            setEmailid(resp.data.email)

          } catch (error) {
            console.log("Not authenticated");
          }
        })();
      }, []);
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
        }, 5000); 
        return () => clearInterval(intervalId);
      }, []);



  return (
    <div><nav class="navbar navbar-expand-lg navbar-dark bg-dark">
    <div class="container-fluid">
      <div class="navbar-brand">
        <a class="stretched-link" href="#">
          <img src="https://boosted.orange.com/docs/5.2/assets/brand/orange-logo.svg" width="50" height="50" alt="Boosted - Back to Home" loading="lazy"/>
        </a>
      </div>
      &nbsp;
   &nbsp;


      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse align-items-end" id="navbarSupportedContent">
        <ul class="navbar-nav me-auto">
       
          <li class="nav-item">
            <a class="nav-link active" aria-current="page" href="/">
            <div> <i class="bi bi-house-door-fill">
            <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" fill="currentColor" class="bi bi-house-door-fill" viewBox="0 0 16 16">
  <path d="M6.5 14.5v-3.505c0-.245.25-.495.5-.495h2c.25 0 .5.25.5.5v3.5a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5v-7a.5.5 0 0 0-.146-.354L13 5.793V2.5a.5.5 0 0 0-.5-.5h-1a.5.5 0 0 0-.5.5v1.293L8.354 1.146a.5.5 0 0 0-.708 0l-6 6A.5.5 0 0 0 1.5 7.5v7a.5.5 0 0 0 .5.5h4a.5.5 0 0 0 .5-.5Z"/>
</svg>
            </i></div>&nbsp;
          <div>Home</div>  </a>
          </li>
          {/* <li class="nav-item">
            <a class="nav-link" href="#">Link</a>
          </li>
          <li class="nav-item dropdown">
            <a class="nav-link dropdown-toggle" href="#" role="button" data-bs-toggle="dropdown" aria-expanded="false">
              Dropdown
            </a>
            <ul class="dropdown-menu dropdown-menu-dark">
              <li><a class="dropdown-item" href="#">Action</a></li>
              <li><a class="dropdown-item" href="#">Another action</a></li>
              <li><hr class="dropdown-divider"/></li>
              <li><a class="dropdown-item" href="#">Something else here</a></li>
            </ul>
          </li> */}
          {/* <li class="nav-item">
            <a class="nav-link disabled">Disabled</a>
          </li> */}
         
          <li>    
   
   
   {/* &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp; */}
   {/* &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp; */}
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp; 
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp; 
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   </li>
 
        </ul>
    

      {user != null && emailid!="admin@gmail.com" ? (
        <div class="d-flex align-items-center">
        <div>
        
 
 <i class="bi bi-currency-dollar"><svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="currentColor" class="bi bi-currency-dollar" viewBox="0 0 16 16">
<path d="M4 10.781c.148 1.667 1.513 2.85 3.591 3.003V15h1.043v-1.216c2.27-.179 3.678-1.438 3.678-3.3 0-1.59-.947-2.51-2.956-3.028l-.722-.187V3.467c1.122.11 1.879.714 2.07 1.616h1.47c-.166-1.6-1.54-2.748-3.54-2.875V1H7.591v1.233c-1.939.23-3.27 1.472-3.27 3.156 0 1.454.966 2.483 2.661 2.917l.61.162v4.031c-1.149-.17-1.94-.8-2.131-1.718H4zm3.391-3.836c-1.043-.263-1.6-.825-1.6-1.616 0-.944.704-1.641 1.8-1.828v3.495l-.2-.05zm1.591 1.872c1.287.323 1.852.859 1.852 1.769 0 1.097-.826 1.828-2.2 1.939V8.73l.348.086z"/>
</svg></i>GAS: {parseInt(gasprice.gas_price)}  GWEI</div>
&nbsp;
&nbsp;
&nbsp;
&nbsp;
{/* &nbsp;
&nbsp;
&nbsp;
&nbsp; */}
<div>
{/* <div>&nbsp;&nbsp;&nbsp;&nbsp;</div> */}
<i class="bi bi-box"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-box" viewBox="0 0 16 16">
  <path d="M8.186 1.113a.5.5 0 0 0-.372 0L1.846 3.5 8 5.961 14.154 3.5 8.186 1.113zM15 4.239l-6.5 2.6v7.922l6.5-2.6V4.24zM7.5 14.762V6.838L1 4.239v7.923l6.5 2.6zM7.443.184a1.5 1.5 0 0 1 1.114 0l7.129 2.852A.5.5 0 0 1 16 3.5v8.662a1 1 0 0 1-.629.928l-7.185 2.874a.5.5 0 0 1-.372 0L.63 13.09a1 1 0 0 1-.63-.928V3.5a.5.5 0 0 1 .314-.464L7.443.184z"/>
</svg></i>&nbsp;
Block: {parseInt(gasprice.current_block_number)}  </div>

&nbsp;
&nbsp;
&nbsp;
&nbsp;
{/* &nbsp;
&nbsp;
&nbsp;
&nbsp; */}

   {/* < div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</div> */}
   <div>

<a href="/Account" style={{ display: 'flex', alignItems: 'center' }}>

  <i class="bi bi-person-circle">
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="29" fill="currentColor" class="bi bi-person-circle" viewBox="0 0 16 16">
      <path d="M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0z"/>
      <path fill-rule="evenodd" d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm8-7a7 7 0 0 0-5.468 11.37C3.242 11.226 4.805 10 8 10s4.757 1.225 5.468 2.37A7 7 0 0 0 8 1z"/>
    </svg>
  </i>
  
  <span>
  <span style={{ color: 'black' }}>a</span>
    <h6 style={{ marginLeft: '10px' }}>{user.username}</h6>
    
  </span>
</a>

{/* <a href="/Account" style={{ display: 'flex', alignItems: 'center' }}>
  <i class="bi bi-person-circle">
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="29" fill="currentColor" class="bi bi-person-circle" viewBox="0 0 16 16">
      <path d="M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0z"/>
      <path fill-rule="evenodd" d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm8-7a7 7 0 0 0-5.468 11.37C3.242 11.226 4.805 10 8 10s4.757 1.225 5.468 2.37A7 7 0 0 0 8 1z"/>
    </svg>
  </i>
  <h6 style={{ marginLeft: '5px' }}>{user.username}</h6>
</a> */}



    {/* <a href="/Account">
  <ul>
    <li>
      <i class="bi bi-person-circle">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="29" fill="currentColor" class="bi bi-person-circle" viewBox="0 0 16 16">
          <path d="M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0z"/>
          <path fill-rule="evenodd" d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm8-7a7 7 0 0 0-5.468 11.37C3.242 11.226 4.805 10 8 10s4.757 1.225 5.468 2.37A7 7 0 0 0 8 1z"/>
        </svg>
      </i>
      <span>
        <h6>{user.username}</h6>
      </span>
    </li>
  </ul>
</a> */}

  
    </div> 
    
  <div>
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;

  </div>
  <div class="row">
  <div class="col md-2">
  <form class="container-fluid justify-content-start align-items-center">
    <div>
      <button class="btn btn-sm btn-primary btn-inverse" type="button" onClick={logoutUser}  >
      <i class="bi bi-box-arrow-left">
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-box-arrow-left" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M6 12.5a.5.5 0 0 0 .5.5h8a.5.5 0 0 0 .5-.5v-9a.5.5 0 0 0-.5-.5h-8a.5.5 0 0 0-.5.5v2a.5.5 0 0 1-1 0v-2A1.5 1.5 0 0 1 6.5 2h8A1.5 1.5 0 0 1 16 3.5v9a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 5 12.5v-2a.5.5 0 0 1 1 0v2z"/>
  <path fill-rule="evenodd" d="M.146 8.354a.5.5 0 0 1 0-.708l3-3a.5.5 0 1 1 .708.708L1.707 7.5H10.5a.5.5 0 0 1 0 1H1.707l2.147 2.146a.5.5 0 0 1-.708.708l-3-3z"/>
</svg>
    </i>   &nbsp; Logout</button>
    </div>
  </form>
  </div>

  <div class="col md-6"> <i class="bi bi-clock"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
  <path d="M8 3.5a.5.5 0 0 0-1 0V9a.5.5 0 0 0 .252.434l3.5 2a.5.5 0 0 0 .496-.868L8 8.71V3.5z"/>
  <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm7-8A7 7 0 1 1 1 8a7 7 0 0 1 14 0z"/>
</svg></i> {formattedTimestamp}</div>
  </div>

</div>


      ) : user != null && emailid === "admin@gmail.com" ? (
        <div class="d-flex align-items-center">
        <div>
        
 
 <i class="bi bi-currency-dollar"><svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="currentColor" class="bi bi-currency-dollar" viewBox="0 0 16 16">
<path d="M4 10.781c.148 1.667 1.513 2.85 3.591 3.003V15h1.043v-1.216c2.27-.179 3.678-1.438 3.678-3.3 0-1.59-.947-2.51-2.956-3.028l-.722-.187V3.467c1.122.11 1.879.714 2.07 1.616h1.47c-.166-1.6-1.54-2.748-3.54-2.875V1H7.591v1.233c-1.939.23-3.27 1.472-3.27 3.156 0 1.454.966 2.483 2.661 2.917l.61.162v4.031c-1.149-.17-1.94-.8-2.131-1.718H4zm3.391-3.836c-1.043-.263-1.6-.825-1.6-1.616 0-.944.704-1.641 1.8-1.828v3.495l-.2-.05zm1.591 1.872c1.287.323 1.852.859 1.852 1.769 0 1.097-.826 1.828-2.2 1.939V8.73l.348.086z"/>
</svg></i>GAS: {parseInt(gasprice.gas_price)}  GWEI</div>
&nbsp;
&nbsp;
&nbsp;
&nbsp;
{/* &nbsp;
&nbsp;
&nbsp;
&nbsp; */}
<div>
{/* <div>&nbsp;&nbsp;&nbsp;&nbsp;</div> */}
<i class="bi bi-box"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-box" viewBox="0 0 16 16">
  <path d="M8.186 1.113a.5.5 0 0 0-.372 0L1.846 3.5 8 5.961 14.154 3.5 8.186 1.113zM15 4.239l-6.5 2.6v7.922l6.5-2.6V4.24zM7.5 14.762V6.838L1 4.239v7.923l6.5 2.6zM7.443.184a1.5 1.5 0 0 1 1.114 0l7.129 2.852A.5.5 0 0 1 16 3.5v8.662a1 1 0 0 1-.629.928l-7.185 2.874a.5.5 0 0 1-.372 0L.63 13.09a1 1 0 0 1-.63-.928V3.5a.5.5 0 0 1 .314-.464L7.443.184z"/>
</svg></i>&nbsp;
Block: {parseInt(gasprice.current_block_number)}  </div>

&nbsp;
&nbsp;
&nbsp;
&nbsp;
{/* &nbsp;
&nbsp;
&nbsp;
&nbsp; */}

       {/* < div>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</div> */}
       <div>
    
      
       {/* <a  href="/Admin">
       <ul>
        <i class="bi bi-person-circle"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="29" fill="currentColor" class="bi bi-person-circle" viewBox="0 0 16 16">
        <path d="M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0z"/>
        <path fill-rule="evenodd" d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm8-7a7 7 0 0 0-5.468 11.37C3.242 11.226 4.805 10 8 10s4.757 1.225 5.468 2.37A7 7 0 0 0 8 1z"/>
        </svg>
        </i>
        <i>
        <h6>{user.username}</h6>
        </i>
        </ul>
        </a> */}



        <a href="/Admin" style={{ display: 'flex', alignItems: 'center' }}>

<i class="bi bi-person-circle">
  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="29" fill="currentColor" class="bi bi-person-circle" viewBox="0 0 16 16">
    <path d="M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0z"/>
    <path fill-rule="evenodd" d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8zm8-7a7 7 0 0 0-5.468 11.37C3.242 11.226 4.805 10 8 10s4.757 1.225 5.468 2.37A7 7 0 0 0 8 1z"/>
  </svg>
</i>

<span>
<span style={{ color: 'black' }}>a</span>
  <h6 style={{ marginLeft: '10px' }}>{user.username}</h6>
  
</span>
</a>













      
        </div> 
        
      <div>
       &nbsp;
       &nbsp;
       &nbsp;
       &nbsp;
       &nbsp;
    
      </div>
      <div class="row">
      <div class="col md-2">
      <form class="container-fluid justify-content-start align-items-center">
        <div>
          <button class="btn btn-sm btn-primary btn-inverse" type="button" onClick={logoutUser}  >
          <i class="bi bi-box-arrow-left">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-box-arrow-left" viewBox="0 0 16 16">
      <path fill-rule="evenodd" d="M6 12.5a.5.5 0 0 0 .5.5h8a.5.5 0 0 0 .5-.5v-9a.5.5 0 0 0-.5-.5h-8a.5.5 0 0 0-.5.5v2a.5.5 0 0 1-1 0v-2A1.5 1.5 0 0 1 6.5 2h8A1.5 1.5 0 0 1 16 3.5v9a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 5 12.5v-2a.5.5 0 0 1 1 0v2z"/>
      <path fill-rule="evenodd" d="M.146 8.354a.5.5 0 0 1 0-.708l3-3a.5.5 0 1 1 .708.708L1.707 7.5H10.5a.5.5 0 0 1 0 1H1.707l2.147 2.146a.5.5 0 0 1-.708.708l-3-3z"/>
    </svg>
        </i>   &nbsp; Logout</button>
        </div>
      </form>
      </div>
    
      <div class="col md-6"> <i class="bi bi-clock"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
  <path d="M8 3.5a.5.5 0 0 0-1 0V9a.5.5 0 0 0 .252.434l3.5 2a.5.5 0 0 0 .496-.868L8 8.71V3.5z"/>
  <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm7-8A7 7 0 1 1 1 8a7 7 0 0 1 14 0z"/>
</svg></i> {formattedTimestamp}</div>
      </div>
    
    </div>
      ) : (
        <div class="d-flex align-items-center">
             <div>
             
      
      <i class="bi bi-currency-dollar"><svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="currentColor" class="bi bi-currency-dollar" viewBox="0 0 16 16">
    <path d="M4 10.781c.148 1.667 1.513 2.85 3.591 3.003V15h1.043v-1.216c2.27-.179 3.678-1.438 3.678-3.3 0-1.59-.947-2.51-2.956-3.028l-.722-.187V3.467c1.122.11 1.879.714 2.07 1.616h1.47c-.166-1.6-1.54-2.748-3.54-2.875V1H7.591v1.233c-1.939.23-3.27 1.472-3.27 3.156 0 1.454.966 2.483 2.661 2.917l.61.162v4.031c-1.149-.17-1.94-.8-2.131-1.718H4zm3.391-3.836c-1.043-.263-1.6-.825-1.6-1.616 0-.944.704-1.641 1.8-1.828v3.495l-.2-.05zm1.591 1.872c1.287.323 1.852.859 1.852 1.769 0 1.097-.826 1.828-2.2 1.939V8.73l.348.086z"/>
  </svg></i>GAS: {parseInt(gasprice.gas_price)}  GWEI</div>
  &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   {/* &nbsp;
   &nbsp;
   &nbsp;
   &nbsp; */}
  <div>
    {/* <div>&nbsp;&nbsp;&nbsp;&nbsp;</div> */}
    <i class="bi bi-box"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-box" viewBox="0 0 16 16">
  <path d="M8.186 1.113a.5.5 0 0 0-.372 0L1.846 3.5 8 5.961 14.154 3.5 8.186 1.113zM15 4.239l-6.5 2.6v7.922l6.5-2.6V4.24zM7.5 14.762V6.838L1 4.239v7.923l6.5 2.6zM7.443.184a1.5 1.5 0 0 1 1.114 0l7.129 2.852A.5.5 0 0 1 16 3.5v8.662a1 1 0 0 1-.629.928l-7.185 2.874a.5.5 0 0 1-.372 0L.63 13.09a1 1 0 0 1-.63-.928V3.5a.5.5 0 0 1 .314-.464L7.443.184z"/>
</svg></i>&nbsp;
  Block: {parseInt(gasprice.current_block_number)} </div>

  &nbsp;
   &nbsp;
   &nbsp;
   &nbsp;
   {/* &nbsp;
   &nbsp;
   &nbsp;
   &nbsp; */}

  <div class="row">
  <div class="col md-12">
      <form class="container-fluid justify-content-start align-items-center">
    <a  href="/login">
    <button class="btn btn-sm btn-primary btn-inverse" type="button"><i class="bi bi-box-arrow-in-right"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-box-arrow-in-right" viewBox="0 0 16 16">
  <path fill-rule="evenodd" d="M6 3.5a.5.5 0 0 1 .5-.5h8a.5.5 0 0 1 .5.5v9a.5.5 0 0 1-.5.5h-8a.5.5 0 0 1-.5-.5v-2a.5.5 0 0 0-1 0v2A1.5 1.5 0 0 0 6.5 14h8a1.5 1.5 0 0 0 1.5-1.5v-9A1.5 1.5 0 0 0 14.5 2h-8A1.5 1.5 0 0 0 5 3.5v2a.5.5 0 0 0 1 0v-2z"/>
  <path fill-rule="evenodd" d="M11.854 8.354a.5.5 0 0 0 0-.708l-3-3a.5.5 0 1 0-.708.708L10.293 7.5H1.5a.5.5 0 0 0 0 1h8.793l-2.147 2.146a.5.5 0 0 0 .708.708l3-3z"/>
</svg></i>&nbsp; Login </button>

   </a>
   &nbsp;
   &nbsp;
   
   <a href="/register" >
    <button class="btn btn-sm btn-primary btn-inverse" type="button"><i class="bi bi-person-add"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-person-add" viewBox="0 0 16 16">
  <path d="M12.5 16a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Zm.5-5v1h1a.5.5 0 0 1 0 1h-1v1a.5.5 0 0 1-1 0v-1h-1a.5.5 0 0 1 0-1h1v-1a.5.5 0 0 1 1 0Zm-2-6a3 3 0 1 1-6 0 3 3 0 0 1 6 0ZM8 7a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z"/>
  <path d="M8.256 14a4.474 4.474 0 0 1-.229-1.004H3c.001-.246.154-.986.832-1.664C4.484 10.68 5.711 10 8 10c.26 0 .507.009.74.025.226-.341.496-.65.804-.918C9.077 9.038 8.564 9 8 9c-5 0-6 3-6 4s1 1 1 1h5.256Z"/>
</svg></i>&nbsp;Sign Up</button>
    </a>
    
  </form>

  </div>
  &nbsp;
   &nbsp;
   {/* */}
<div class="col md-6"> <i class="bi bi-clock"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" class="bi bi-clock" viewBox="0 0 16 16">
  <path d="M8 3.5a.5.5 0 0 0-1 0V9a.5.5 0 0 0 .252.434l3.5 2a.5.5 0 0 0 .496-.868L8 8.71V3.5z"/>
  <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zm7-8A7 7 0 1 1 1 8a7 7 0 0 1 14 0z"/>
</svg></i> {formattedTimestamp}</div>
   
 
  </div>
  </div>
     )}



 
  </div> 
    </div>
    
    
  </nav>

  </div>
  )
}
export default Header;
