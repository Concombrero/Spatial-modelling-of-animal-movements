//-------------------------------------
// Document options
//

#let option = (
  type : none,
  lang : "en",
)
//-------------------------------------
// Optional generate titlepage image
//
// Helper function for unnumbered headers
#let nonumber(body) = {
  set heading(numbering: none)
  body
}


//-------------------------------------
// Metadata of the document
//
#let doc= (
  title    : [*[Place Holder]*],
  url      : "",
  logos: (
    tp_topleft  : image("assets/ua_logo_green_rgb.png", width: 80%),
    tp_topright : "",
    tp_main     : "",
  ),
  authors: (
    (
      name        : "BOYER Timothé",
      abbr        : none,
      email       : "",
    ),
 ),
  school: (
    name        : "University of Alberta",
  ),

  keywords : ("keyword1", "keyword2", "keyword3"),)

#let date= datetime.today()

//-------------------------------------
// Settings
//
#let tableof = (
  toc: false,
  tof: false,
  tot: false,
  tol: false,
  toe: false,
  maxdepth: 3,
)

#let gloss    = false
#let appendix = false
#let bib = (
  display : true,
  path  : "/assets/bibliography.bib",
  style : "ieee",
)
