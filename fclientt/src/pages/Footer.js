// import React from 'react'

// export const Footer = () => {
//   return (
//     <div><footer class="footer bg-dark navbar-dark">
//     <h2 class="visually-hidden">Sitemap & information</h2>
//     <div class="container-xxl footer-terms">
//       <ul class="navbar-nav gap-md-3">
//         <li class="fw-bold">© Orange 2023</li>
//         <li><a class="nav-link" href="#">Terms and conditions</a></li>
//         <li><a class="nav-link" href="#">Privacy</a></li>
//         <li><a class="nav-link" href="#">Accessibility statement</a></li>
//         <li><a class="nav-link" href="#">Cookie policy</a></li>
//       </ul>
//     </div>
//   </footer></div>
//   )
// }
// export default Footer;
import React from 'react';

const Footer = () => {
  return (
    <footer className="footer bg-dark navbar-dark fixed-bottom">
      <h2 className="visually-hidden">Sitemap & information</h2>
      <div className="container-xxl footer-terms">
        <ul className="navbar-nav gap-md-3">
          <li className="fw-bold">© Orange 2023</li>
          <li><a className="nav-link" href="#">Terms and conditions</a></li>
          <li><a className="nav-link" href="#">Privacy</a></li>
          <li><a className="nav-link" href="#">Accessibility statement</a></li>
          <li><a className="nav-link" href="#">Cookie policy</a></li>
        </ul>
      </div>
    </footer>
  );
};

export default Footer;
