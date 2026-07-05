import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { PageList } from "./PageList"

const AllNotes: QuartzComponent = (props: QuartzComponentProps) => {
  const pages = props.allFiles.filter((page) => page.slug !== "index")

  return (
    <div class="all-notes page-listing">
      <h2>Shelf</h2>
      <PageList {...props} allFiles={pages} showDescription />
    </div>
  )
}

AllNotes.css = `
${PageList.css}

.all-notes > h2 {
  margin: 0 0 1rem;
}
`

export default (() => AllNotes) satisfies QuartzComponentConstructor
