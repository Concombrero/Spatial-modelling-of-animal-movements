#import "@preview/hei-synd-report:0.1.1": *
#import "metadata.typ": *
#import "extra.typ": *
//#show:make-glossary
//#register-glossary(entry-list)

//-------------------------------------
// Template config
//
#show: report.with(
  option: option,
  doc: doc,
  date: date,
  tableof: tableof,
)

//-------------------------------------
// Content
//
#counter(page).update(1)

#let project-heading-numbering(..nums) = {
  let ns = nums.pos()
  if ns.len() == 1 {
    numbering("1.", ns.at(0))
  } else if ns.len() == 2 {
    numbering("a)", ns.at(1))
  } else {
    numbering("a.1", ns.at(1), ns.at(2))
  }
}

#set heading(numbering: project-heading-numbering)


= Abstract


= Introduction

